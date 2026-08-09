import gymnasium as gym
import numpy as np
import hisss

# Safe import: Works with Ray during training, and without Ray on EC2/CI
try:
    from ray.rllib.env.multi_agent_env import MultiAgentEnv
except ModuleNotFoundError:
    class MultiAgentEnv(gym.Env):
        """Fallback dummy class when Ray is not installed on lightweight deployment instances."""
        pass

class BattlesnakeBlackoutEnv(MultiAgentEnv):
    def __init__(self, config=None):
        super().__init__()
        self.config = config or {}
        
        self.game_config = hisss.standard_config()
        try: 
            self.game_config.num_players = 4
            self.game_config.all_actions_legal = True 
            self.game_config.w = 15
            self.game_config.h = 15
            self.game_config.food_spawn_chance = 15  
            self.game_config.min_food = 4 # Standard 4-player Battlesnake food density
        except Exception: pass
            
        self.env = hisss.BattleSnakeGame(self.game_config)
        self.num_snakes = 4
        
        self.agent_ids = [f"snake_{i}" for i in range(self.num_snakes)]
        self._agent_ids = set(self.agent_ids)
        
        single_action_space = gym.spaces.Discrete(4)
        
        # Explicit float32 space, bounds set to infinity to bypass Ray strict Box checks
        single_agent_obs_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(11, 11, 22), dtype=np.float32)
        
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

        # Pre-calculate Diamond Mask as float32
        y, x = np.ogrid[:11, :11]
        mask_2d = (abs(x - 5) + abs(y - 5) <= 5).astype(np.float32)
        self.diamond_mask = mask_2d[:, :, np.newaxis]

    def _get_unpacked_obs(self):
        obs_data, _, _ = self.env.get_obs()
        alive_ids = self.env.players_alive()
        
        if isinstance(obs_data, dict):
            actor_obs_array = obs_data["actor_obs"]
        else:
            actor_obs_array = obs_data 
        
        unpacked = {}
        
        for i in range(self.num_snakes):
            agent_id = f"snake_{i}"
            
            if i in alive_ids:
                idx = alive_ids.index(i)
                raw_actor_obs = actor_obs_array[idx].copy().astype(np.float32)
                
                # --- PER-CHANNEL NORMALIZATION ---
                # Some channels (e.g. health/hunger) are encoded 0-255 while others
                # (e.g. body/food/wall flags) are already 0/1. A single global max()
                # check divides the ENTIRE tensor by 255 the moment any one channel
                # is saturated, which crushes the already-correct 0/1 channels down
                # to ~0.004 and makes the board effectively invisible to the network.
                # Normalize each channel independently instead.
                channel_max = raw_actor_obs.max(axis=(0, 1), keepdims=True)
                scale = np.where(channel_max > 1.0, 255.0, 1.0)
                raw_actor_obs = raw_actor_obs / scale
                
                # Inject walls as 1.0 (Lethal boundary)
                out_of_bounds = raw_actor_obs[:, :, 1] == 0
                raw_actor_obs[out_of_bounds, 2] = 1.0  
                raw_actor_obs[out_of_bounds, 13] = 1.0 
                
                # hisss natively centers raw_actor_obs around the snake's head at [14, 14].
                # We crop an 11x11 window centered at index 14 (slice 9 to 20).
                cropped_ego_obs = raw_actor_obs[9:20, 9:20, :]
                
                # Apply the Blackout Diamond Mask
                cropped_ego_obs = cropped_ego_obs * self.diamond_mask
                
                # Force clamp to [0.0, 1.0] to prevent gradient flashbangs & -0.0 artifacts
                cropped_ego_obs = np.clip(cropped_ego_obs, 0.0, 1.0)
                
                unpacked[agent_id] = cropped_ego_obs
            else:
                unpacked[agent_id] = np.zeros((11, 11, 22), dtype=np.float32)
                
        return unpacked

    def reset(self, *, seed=None, options=None):
        self.env.reset()
        self.turn_count = 0
        self.last_actions = {i: 0 for i in range(self.num_snakes)}
        self.terminated_agents = set()
        self.previous_lengths = {f"snake_{i}": 3 for i in range(self.num_snakes)} 
        infos = {agent_id: {} for agent_id in self.agent_ids}
        return self._get_unpacked_obs(), infos

    def step(self, action_dict):
        self.turn_count += 1
        alive_ids = self.env.players_alive()
        pre_step_state = self.env.get_state()
        
        head_coords_pre_step = {}
        body_coords_pre_step = {}
        lengths_pre_step = {}
        health_pre_step = {} 
        
        for i in alive_ids:
            agent_id = f"snake_{i}"
            snake_body = pre_step_state.snake_pos[i] 
            head_coords_pre_step[agent_id] = snake_body[0]
            body_coords_pre_step[agent_id] = snake_body 
            lengths_pre_step[agent_id] = int(pre_step_state.snake_len[i])
            health_pre_step[agent_id] = int(pre_step_state.snake_health[i])

        actions = []
        penalty_flags = {i: False for i in range(self.num_snakes)}
        opposites = {0: 2, 2: 0, 1: 3, 3: 1}
        action_vecs = {0: (0, 1), 1: (1, 0), 2: (0, -1), 3: (-1, 0)}
        intended_heads = {}
        
        for i in alive_ids:
            action = int(action_dict.get(f"snake_{i}", 0))
            if self.turn_count > 1:
                last_act = self.last_actions[i]
                if action == opposites.get(last_act):
                    action = last_act  
                    penalty_flags[i] = True
            actions.append(action)
            self.last_actions[i] = action

            hx, hy = head_coords_pre_step[f"snake_{i}"]
            dx, dy = action_vecs[action]
            intended_heads[i] = (hx + dx, hy + dy)

        try:
            raw_rewards, done, _ = self.env.step(tuple(actions))
        except ValueError:
            raw_rewards, done = [-2.0]*len(alive_ids), True

        current_alive = self.env.players_alive()
        game_over = bool(done) or len(current_alive) <= 1
        died_this_turn = [f"snake_{snake}" for snake in alive_ids if snake not in current_alive]

        obs, rewards, infos = {}, {}, {}
        terminations, truncations = {"__all__": game_over}, {"__all__": False}

        current_state = self.env.get_state()

        if not game_over:
            obs_unpacked = self._get_unpacked_obs()

        for i in range(self.num_snakes):
            agent_id = f"snake_{i}"
            if agent_id in self.terminated_agents:
                continue

            if game_over:
                obs[agent_id] = np.zeros((11, 11, 22), dtype=np.float32)
            else:
                obs[agent_id] = obs_unpacked[agent_id]

            is_dead = i not in current_alive
            terminations[agent_id] = game_over or is_dead
            truncations[agent_id] = False
            infos[agent_id] = {}

            raw = float(raw_rewards[alive_ids.index(i)]) if i in alive_ids else 0.0
            my_length = lengths_pre_step.get(agent_id, 3)

            head_on_partners = [
                other_id for other_id in alive_ids 
                if other_id != i and intended_heads.get(i) == intended_heads.get(other_id)
            ]

            if is_dead or raw < 0:
                players_alive_at_start = len(alive_ids)
                if players_alive_at_start >= 4: base_death = -15.0  
                elif players_alive_at_start == 3: base_death = -10.0  
                elif players_alive_at_start == 2: base_death = -3.0   
                else: base_death = 0.0
                
                death_penalty = min(0.0, base_death)

                for other_id in head_on_partners:
                    other_len = lengths_pre_step.get(f"snake_{other_id}", 3)
                    if my_length <= other_len:
                        death_penalty -= 10.0
                        break

                rewards[agent_id] = death_penalty
            else:
                step_reward = -2.0 if penalty_flags[i] else 0.0
                rewards[agent_id] = step_reward
                
                current_true_length = int(current_state.snake_len[i])
                ate_food = current_true_length > self.previous_lengths.get(agent_id, 3)
                
                if ate_food:
                    my_pre_health = health_pre_step.get(agent_id, 100)
                    hunger_multiplier = (100.0 - float(my_pre_health)) / 100.0
                    rewards[agent_id] += 1.0 + (6.0 * hunger_multiplier)
                    self.previous_lengths[agent_id] = current_true_length

                for other_id in current_alive:
                    if other_id != i:
                        other_next = intended_heads.get(other_id)
                        my_next = intended_heads.get(i)
                        other_len = lengths_pre_step.get(f"snake_{other_id}", 3)
                        if my_next and other_next and my_length <= other_len:
                            next_dist = abs(my_next[0] - other_next[0]) + abs(my_next[1] - other_next[1])
                            if next_dist <= 1:
                                rewards[agent_id] -= 0.75 
                    
                for dead_id in died_this_turn:
                    if dead_id != agent_id:
                        dead_int = int(dead_id.split("_")[1])
                        
                        if dead_int in head_on_partners and my_length > lengths_pre_step.get(dead_id, 3):
                            rewards[agent_id] += 5.0
                        else:
                            dead_next = intended_heads.get(dead_int)
                            my_body = body_coords_pre_step.get(agent_id, [])
                            if dead_next and my_body:
                                effective_body = my_body if ate_food else my_body[:-1]
                                is_trap = any(seg[0] == dead_next[0] and seg[1] == dead_next[1] for seg in effective_body)
                                if is_trap: 
                                    rewards[agent_id] += 7.0
                                    
                if game_over:
                    rewards[agent_id] += 15.0

            if terminations[agent_id]:
                self.terminated_agents.add(agent_id)
        
        return obs, rewards, terminations, truncations, infos
    
    def render(self):
        self.env.render()