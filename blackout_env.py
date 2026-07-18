import gymnasium as gym
import numpy as np
from ray.rllib.env.multi_agent_env import MultiAgentEnv
import hisss

class BattlesnakeBlackoutEnv(MultiAgentEnv):
    def __init__(self, config=None):
        super().__init__()
        self.config = config or {}
        
        self.game_config = hisss.standard_config()
        try: 
            self.game_config.num_players = 4
            self.game_config.all_actions_legal = True 
            
            # Use the official 15x15 board size
            self.game_config.w = 15
            self.game_config.h = 15
            
            # Return to official tournament food settings!
            self.game_config.food_spawn_chance = 15  
            self.game_config.min_food = 1           
        except Exception: pass
            
        self.env = hisss.BattleSnakeGame(self.game_config)
        self.num_snakes = 4
        
        self.agent_ids = [f"snake_{i}" for i in range(self.num_snakes)]
        self._agent_ids = set(self.agent_ids)
        
        single_action_space = gym.spaces.Discrete(4)
        single_agent_obs_space = gym.spaces.Dict({
            "obs": gym.spaces.Box(low=0, high=255, shape=(29, 29, 22), dtype=np.uint8),
            "state": gym.spaces.Box(low=0, high=255, shape=(29, 29, 22), dtype=np.uint8)
        })
        
        self.action_space = gym.spaces.Dict({
            agent_id: single_action_space for agent_id in self.agent_ids
        })
        self.observation_space = gym.spaces.Dict({
            agent_id: single_agent_obs_space for agent_id in self.agent_ids
        })
        
        self.last_actions = {i: 0 for i in range(self.num_snakes)}
        self.turn_count = 0
        self.terminated_agents = set() 
        self.previous_lengths = {}

    def _get_unpacked_obs(self):
        obs_dict, _, _ = self.env.get_obs()
        alive_ids = self.env.players_alive()
        
        # Grab the C++ state so we can find exactly where every snake's head is
        current_state = self.env.get_state()
        
        unpacked = {}
        for i in range(self.num_snakes):
            agent_id = f"snake_{i}"
            
            if i in alive_ids:
                idx = alive_ids.index(i)
                raw_actor_obs = obs_dict["actor_obs"][idx]
                raw_critic_obs = obs_dict["critic_obs"][idx]
                
                # --- APPLY 5-SQUARE BLACKOUT ---
                # 1. Get exact head coordinate from C++ state
                head_x, head_y = current_state.snake_pos[i][0]
                
                # 2. Shift coordinates to account for the 29x29 padding
                # (A 15x15 board centered in 29x29 means a border offset of 7)
                tensor_x = head_x + 7
                tensor_y = head_y + 7
                
                # 3. Create a totally black canvas
                masked_actor_obs = np.zeros_like(raw_actor_obs)
                
                # 4. Define the 5-square radius bounds
                view_radius = 5
                min_x = max(0, tensor_x - view_radius)
                max_x = min(29, tensor_x + view_radius + 1)
                
                min_y = max(0, tensor_y - view_radius)
                max_y = min(29, tensor_y + view_radius + 1)
                
                # 5. Copy ONLY the visible area into the black canvas
                masked_actor_obs[min_x:max_x, min_y:max_y, :] = raw_actor_obs[min_x:max_x, min_y:max_y, :]
                
                unpacked[agent_id] = {
                    "obs": masked_actor_obs,   # Actor sees Fog of War
                    "state": raw_critic_obs    # Critic sees the whole board!
                }
            else:
                # Dead snakes see nothing
                unpacked[agent_id] = {
                    "obs": np.zeros((29, 29, 22), dtype=np.uint8), 
                    "state": np.zeros((29, 29, 22), dtype=np.uint8)
                }
                
        return unpacked

    def reset(self, *, seed=None, options=None):
        self.env.reset()
        self.turn_count = 0
        self.last_actions = {i: 0 for i in range(self.num_snakes)}
        self.terminated_agents = set()
        
        # We only need to track previous lengths to calculate the food delta (+3.0)
        self.previous_lengths = {f"snake_{i}": 3 for i in range(self.num_snakes)} 
        
        infos = {agent_id: {} for agent_id in self.agent_ids}
        return self._get_unpacked_obs(), infos

    def step(self, action_dict):
        self.turn_count += 1
        
        alive_ids = self.env.players_alive()
        alive_count_start = len(alive_ids)
        
        # ---------------------------------------------------------
        # PRE-STEP: Grab exact coordinates and lengths via the C++ State
        # ---------------------------------------------------------
        pre_step_state = self.env.get_state()
        
        head_coords_pre_step = {}
        body_coords_pre_step = {}
        lengths_pre_step = {}
        
        for i in alive_ids:
            agent_id = f"snake_{i}"
            # snake_pos contains the full body array. Index 0 is the head.
            snake_body = pre_step_state.snake_pos[i] 
            
            head_coords_pre_step[agent_id] = snake_body[0]
            body_coords_pre_step[agent_id] = snake_body # Save the whole body
            lengths_pre_step[agent_id] = int(pre_step_state.snake_len[i])

        # ---------------------------------------------------------
        # EXECUTE STEP
        # ---------------------------------------------------------
        actions = []
        penalty_flags = {i: False for i in range(self.num_snakes)}
        opposites = {0: 2, 2: 0, 1: 3, 3: 1}
        
        for i in alive_ids:
            action = int(action_dict.get(f"snake_{i}", 0))
            if self.turn_count > 1:
                last_act = self.last_actions[i]
                if action == opposites.get(last_act):
                    action = last_act  # Keep moving straight instead of dying!
                    penalty_flags[i] = True
                    
            actions.append(action)
            self.last_actions[i] = action

        try:
            raw_rewards, done, _ = self.env.step(tuple(actions))
        except ValueError:
            raw_rewards, done = [-2.0]*len(alive_ids), True

        current_alive = self.env.players_alive()
        game_over = bool(done) or len(current_alive) <= 1
        died_this_turn = [f"snake_{snake}" for snake in alive_ids if snake not in current_alive]

        obs, rewards, infos = {}, {}, {}
        terminations, truncations = {"__all__": game_over}, {"__all__": False}

        if not game_over:
            obs_unpacked = self._get_unpacked_obs()
            current_state = self.env.get_state()

        # ---------------------------------------------------------
        # REWARD ALLOCATION
        # ---------------------------------------------------------
        for i in range(self.num_snakes):
            agent_id = f"snake_{i}"
            
            if agent_id in self.terminated_agents:
                continue

            if game_over:
                obs[agent_id] = {
                    "obs": np.zeros((29, 29, 22), dtype=np.uint8), 
                    "state": np.zeros((29, 29, 22), dtype=np.uint8)
                }
            else:
                obs[agent_id] = obs_unpacked[agent_id]

            is_dead = i not in current_alive
            terminations[agent_id] = game_over or is_dead
            truncations[agent_id] = False
            infos[agent_id] = {}

            raw = float(raw_rewards[alive_ids.index(i)]) if i in alive_ids else 0.0
            
            # 1. FIX C: REDUCED DEATH PENALTY (Tiered based on placement)
            if is_dead or raw < 0:
                # Halved from -10.0 to encourage risky plays!
                if alive_count_start >= 4: base_penalty = -5.0
                elif alive_count_start == 3: base_penalty = -3.0
                elif alive_count_start == 2: base_penalty = -1.0
                else: base_penalty = -5.0
                
                # Still reward them for dying massive instead of dying small
                length_bonus = self.previous_lengths.get(agent_id, 3) * 0.5
                rewards[agent_id] = min(base_penalty + length_bonus, -0.5) 
                
            elif game_over:
                rewards[agent_id] = 15.0 
            else:
                # 2. BASE STEP PENALTIES
                step_reward = -2.0 if penalty_flags[i] else 0.0
                rewards[agent_id] = step_reward
                
                # 3. EXACT FOOD REWARD & STARVATION TRACKING
                current_true_length = int(current_state.snake_len[i])
                current_true_health = int(current_state.snake_health[i])
                
                # If they grew, give them points
                if current_true_length > self.previous_lengths.get(agent_id, 3):
                    rewards[agent_id] += 3.0
                    self.previous_lengths[agent_id] = current_true_length
                
                if current_true_health < 30:
                    rewards[agent_id] -= 0.05
                    
                # 4. FIX A: EXACT H2H KILL AND TRAPPING VERIFICATION
                for dead_id in died_this_turn:
                    if dead_id != agent_id:
                        my_head = head_coords_pre_step.get(agent_id)
                        my_body = body_coords_pre_step.get(agent_id, [])
                        dead_head = head_coords_pre_step.get(dead_id)
                        
                        if my_head and dead_head and my_body:
                            my_len = lengths_pre_step.get(agent_id, 0)
                            dead_len = lengths_pre_step.get(dead_id, 0)
                            
                            head_dist = abs(my_head[0] - dead_head[0]) + abs(my_head[1] - dead_head[1])
                            
                            # 4a. EXACT KILL RULE: Heads collided and we were strictly longer
                            if head_dist <= 2 and my_len > dead_len:
                                rewards[agent_id] += 5.0
                            else:
                                # 4b. TRAP / CUT-OFF RULE: Did they die adjacent to my body?
                                # We skip my_body[0] because that's our head (handled above)
                                is_trap = False
                                for segment in my_body[1:]:
                                    body_dist = abs(segment[0] - dead_head[0]) + abs(segment[1] - dead_head[1])
                                    if body_dist <= 1: # They crashed right into our side!
                                        is_trap = True
                                        break
                                
                                if is_trap:
                                    rewards[agent_id] += 7.0

            if terminations[agent_id]:
                self.terminated_agents.add(agent_id)
        
        return obs, rewards, terminations, truncations, infos
    
    def render(self):
        self.env.render()