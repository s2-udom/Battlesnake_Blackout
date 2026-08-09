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
        # ... rest of your environment code remains completely untouched ...
        self.config = config or {}
        
        self.game_config = hisss.standard_config()
        try: 
            self.game_config.num_players = 4
            self.game_config.all_actions_legal = True 
            self.game_config.w = 15
            self.game_config.h = 15
            self.game_config.food_spawn_chance = 15  
            self.game_config.min_food = 1           
        except Exception: pass
            
        self.env = hisss.BattleSnakeGame(self.game_config)
        self.num_snakes = 4
        
        self.agent_ids = [f"snake_{i}" for i in range(self.num_snakes)]
        self._agent_ids = set(self.agent_ids)
        
        single_action_space = gym.spaces.Discrete(4)
        single_agent_obs_space = gym.spaces.Box(low=0, high=255, shape=(11, 11, 22), dtype=np.uint8)
        
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

        # OPTIMIZATION 1: Pre-calculate the Diamond Mask as a 3D multiplier for C-level speed
        y, x = np.ogrid[:11, :11]
        mask_2d = (abs(x - 5) + abs(y - 5) <= 5).astype(np.uint8)
        self.diamond_mask = mask_2d[:, :, np.newaxis] # Shape becomes (11, 11, 1)

    def _get_unpacked_obs(self):
        # 1. Rename to obs_data since it might not be a dictionary anymore
        obs_data, _, _ = self.env.get_obs()
        alive_ids = self.env.players_alive()
        
        # 2. Add this version-check fallback:
        if isinstance(obs_data, dict):
            actor_obs_array = obs_data["actor_obs"]
        else:
            actor_obs_array = obs_data 
        
        # Grab state once to look up exact head coordinates from memory
        current_state = self.env.get_state()
        
        unpacked = {}
        for i in range(self.num_snakes):
            agent_id = f"snake_{i}"
            
            if i in alive_ids:
                idx = alive_ids.index(i)
                
                # 3. Use the safely extracted array here
                raw_actor_obs = actor_obs_array[idx].copy() 
                
                out_of_bounds = raw_actor_obs[:, :, 1] == 0
                raw_actor_obs[out_of_bounds, 2] = 255  
                raw_actor_obs[out_of_bounds, 13] = 255 
                
                # ... the rest of your optimization code remains exactly the same ...

                # OPTIMIZATION 2: Instant head lookup (No more np.argwhere scanning!)
                my_head = current_state.snake_pos[i][0]
                tx, ty = my_head[0], my_head[1]
                
                # The Safe 29x29 Ego-Shift Canvas
                shift_x = 14 - tx
                shift_y = 14 - ty
                
                ego_actor_obs = np.zeros_like(raw_actor_obs)
                
                src_x_min = max(0, -shift_x)
                src_x_max = min(29, 29 - shift_x)
                src_y_min = max(0, -shift_y)
                src_y_max = min(29, 29 - shift_y)
                
                dst_x_min = max(0, shift_x)
                dst_x_max = min(29, 29 + shift_x)
                dst_y_min = max(0, shift_y)
                dst_y_max = min(29, 29 + shift_y)
                
                ego_actor_obs[dst_x_min:dst_x_max, dst_y_min:dst_y_max, :] = \
                    raw_actor_obs[src_x_min:src_x_max, src_y_min:src_y_max, :]
                
                cropped_ego_obs = ego_actor_obs[9:20, 9:20, :]
                
                # OPTIMIZATION 1: Blazing fast contiguous matrix multiplication
                cropped_ego_obs *= self.diamond_mask
                
                unpacked[agent_id] = cropped_ego_obs
            else:
                unpacked[agent_id] = np.zeros((11, 11, 22), dtype=np.uint8)
                
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
        
        for i in alive_ids:
            agent_id = f"snake_{i}"
            snake_body = pre_step_state.snake_pos[i] 
            head_coords_pre_step[agent_id] = snake_body[0]
            body_coords_pre_step[agent_id] = snake_body 
            lengths_pre_step[agent_id] = int(pre_step_state.snake_len[i])

        actions = []
        penalty_flags = {i: False for i in range(self.num_snakes)}
        opposites = {0: 2, 2: 0, 1: 3, 3: 1}
        
        for i in alive_ids:
            action = int(action_dict.get(f"snake_{i}", 0))
            if self.turn_count > 1:
                last_act = self.last_actions[i]
                if action == opposites.get(last_act):
                    action = last_act  
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

        for i in range(self.num_snakes):
            agent_id = f"snake_{i}"
            if agent_id in self.terminated_agents:
                continue

            if game_over:
                obs[agent_id] = np.zeros((11, 11, 22), dtype=np.uint8)
            else:
                obs[agent_id] = obs_unpacked[agent_id]

            is_dead = i not in current_alive
            terminations[agent_id] = game_over or is_dead
            truncations[agent_id] = False
            infos[agent_id] = {}

            raw = float(raw_rewards[alive_ids.index(i)]) if i in alive_ids else 0.0
            my_length = lengths_pre_step.get(agent_id, 3)
            
            if is_dead or raw < 0:
                players_alive_at_start = len(alive_ids)
                if players_alive_at_start >= 4: base_death = -15.0  
                elif players_alive_at_start == 3: base_death = -10.0  
                elif players_alive_at_start == 2: base_death = -3.0   
                else: base_death = 0.0
                rewards[agent_id] = min(0.0, base_death) 
            elif game_over:
                rewards[agent_id] = 15.0 
            else:
                step_reward = -2.0 if penalty_flags[i] else 0.0
                my_head = head_coords_pre_step.get(agent_id)
                if my_head:
                    hx, hy = my_head[0], my_head[1]
                    if hx <= 1 or hx >= 13 or hy <= 1 or hy >= 13: 
                        step_reward -= 0.1
                    
                    # CHANGED: Removed the 'center control' +0.05 bonus entirely.
                    # In Blackout, the center is a death trap.
                    
                    for other_id in alive_ids:
                        if other_id != i:
                            enemy_body = pre_step_state.snake_pos[other_id]
                            # CHANGED: Added [1:-1] to ignore the very last segment (the tail).
                            # This teaches the agent the advanced "tail chasing" maneuver.
                            for segment in enemy_body[1:-1]:
                                dist_to_segment = abs(hx - segment[0]) + abs(hy - segment[1])
                                if dist_to_segment == 1: 
                                    step_reward -= 0.15 
                        
                rewards[agent_id] = step_reward
                
                current_true_length = int(current_state.snake_len[i])
                current_true_health = int(current_state.snake_health[i])
                
                # CHANGED: Dynamic Food Rewards. 
                # Scales the reward based on how starving the snake was before eating.
                if current_true_length > self.previous_lengths.get(agent_id, 3):
                    hunger_multiplier = (100.0 - float(current_true_health)) / 100.0
                    eating_reward = 1.0 + (6.0 * hunger_multiplier)
                    rewards[agent_id] += eating_reward
                    self.previous_lengths[agent_id] = current_true_length
                
                # Health deficit penalty (Already working perfectly, left as-is)
                health_deficit = (100 - current_true_health) * 0.02
                rewards[agent_id] -= health_deficit
                    
                for dead_id in died_this_turn:
                    if dead_id != agent_id:
                        my_body = body_coords_pre_step.get(agent_id, [])
                        dead_head = head_coords_pre_step.get(dead_id)
                        
                        if my_head and dead_head and my_body:
                            dead_len = lengths_pre_step.get(dead_id, 3)
                            head_dist = abs(my_head[0] - dead_head[0]) + abs(my_head[1] - dead_head[1])
                            
                            if head_dist <= 2:
                                if my_length > dead_len:
                                    # We had a length advantage and killed them head-to-head
                                    rewards[agent_id] += 5.0
                                else:
                                    # CHANGED: The 50/50 Gamble Deterrent.
                                    # Penalizes the agent for engaging in equal/smaller head-on collisions.
                                    rewards[agent_id] -= 3.0
                            else:
                                is_trap = False
                                for segment in my_body[1:]:
                                    body_dist = abs(segment[0] - dead_head[0]) + abs(segment[1] - dead_head[1])
                                    if body_dist <= 1: 
                                        is_trap = True
                                        break
                                if is_trap: 
                                    rewards[agent_id] += 7.0

            if terminations[agent_id]:
                self.terminated_agents.add(agent_id)
        
        return obs, rewards, terminations, truncations, infos
    
    def render(self):
        self.env.render()