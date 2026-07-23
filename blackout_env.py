import gymnasium as gym
import numpy as np
import hisss

class BattlesnakeBlackoutEnv:
    def __init__(self, config=None):
        self.config = config or {}
        
        self.game_config = hisss.standard_config()
        try: 
            self.game_config.num_players = 4
            self.game_config.all_actions_legal = True 
            
            # Use the official 15x15 board size
            self.game_config.w = 15
            self.game_config.h = 15
            
            # Return to official tournament food settings
            self.game_config.food_spawn_chance = 15  
            self.game_config.min_food = 1           
            
        except Exception: pass
            
        self.env = hisss.BattleSnakeGame(self.game_config)
        self.num_snakes = 4
        
        self.agent_ids = [f"snake_{i}" for i in range(self.num_snakes)]
        self._agent_ids = set(self.agent_ids)
        
        single_action_space = gym.spaces.Discrete(4)
        
        # PURE 11x11 BOX: No more dictionaries!
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

    def _get_unpacked_obs(self):
        obs_dict, _, _ = self.env.get_obs()
        alive_ids = self.env.players_alive()
        
        unpacked = {}
        for i in range(self.num_snakes):
            agent_id = f"snake_{i}"
            
            if i in alive_ids:
                idx = alive_ids.index(i)
                raw_actor_obs = obs_dict["actor_obs"][idx].copy() # Copy to avoid mutating shared state
                
                # --- THE FIX: PAINT THE WALLS IN TRAINING ---
                # Find all tiles where "Valid Board" (Channel 1) is 0.0, and mark them as Hazards (Ch 2)
                # and Enemy Bodies (Ch 13)
                out_of_bounds = raw_actor_obs[:, :, 1] == 0
                raw_actor_obs[out_of_bounds, 2] = 255  # Hazard
                raw_actor_obs[out_of_bounds, 13] = 255 # Enemy Body
                # --------------------------------------------

                # 1. FIND THE HEAD: Scan Channel 6 to find exactly where hisss put the head
                head_locs = np.argwhere(raw_actor_obs[:, :, 6] > 0)
                if len(head_locs) > 0:
                    tx, ty = head_locs[0]
                else:
                    tx, ty = 14, 14 # Fallback
                
                # 2. THE EGO-SHIFT: Calculate how far we must move the board to center the head
                shift_x = 14 - tx
                shift_y = 14 - ty
                
                ego_actor_obs = np.zeros_like(raw_actor_obs)
                
                # Safely copy the board to the new shifted coordinates (preventing wraparound)
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
                
                # 3. APPLY THE FOG AND CROP: Slice out the 11x11 window and return it directly
                cropped_ego_obs = ego_actor_obs[9:20, 9:20, :]
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

        # ---------------------------------------------------------
        # THE ELITE REWARD ALLOCATION (Tiered Hunter Update)
        # ---------------------------------------------------------
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
                
                if players_alive_at_start >= 4:
                    base_death = -15.0  
                elif players_alive_at_start == 3:
                    base_death = -10.0  
                elif players_alive_at_start == 2:
                    base_death = -3.0   
                else:
                    base_death = 0.0
                    
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
                    elif 4 <= hx <= 10 and 4 <= hy <= 10:
                        step_reward += 0.05
                        
                    for other_id in alive_ids:
                        if other_id != i:
                            enemy_body = pre_step_state.snake_pos[other_id]
                            for segment in enemy_body[1:]:
                                dist_to_segment = abs(hx - segment[0]) + abs(hy - segment[1])
                                if dist_to_segment == 1:
                                    step_reward -= 0.15 
                        
                rewards[agent_id] = step_reward
                
                current_true_length = int(current_state.snake_len[i])
                current_true_health = int(current_state.snake_health[i])
                
                if current_true_length > self.previous_lengths.get(agent_id, 3):
                    rewards[agent_id] += 3.0
                    self.previous_lengths[agent_id] = current_true_length
                
                if current_true_health < 50:
                    panic_factor = (50 - current_true_health) * 0.02
                    rewards[agent_id] -= panic_factor
                    
                for dead_id in died_this_turn:
                    if dead_id != agent_id:
                        my_body = body_coords_pre_step.get(agent_id, [])
                        dead_head = head_coords_pre_step.get(dead_id)
                        
                        if my_head and dead_head and my_body:
                            dead_len = lengths_pre_step.get(dead_id, 3)
                            head_dist = abs(my_head[0] - dead_head[0]) + abs(my_head[1] - dead_head[1])
                            
                            if head_dist <= 2 and my_length > dead_len:
                                rewards[agent_id] += 5.0
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