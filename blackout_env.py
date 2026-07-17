import gymnasium as gym
import numpy as np
from ray.rllib.env.multi_agent_env import MultiAgentEnv
import hisss

class BattlesnakeBlackoutEnv(MultiAgentEnv):
    def __init__(self, config=None):
        super().__init__()
        self.config = config or {}
        
        # 1. Start the 4-player chaos
        self.game_config = hisss.standard_config()
        try: 
            self.game_config.num_players = 4
            self.game_config.all_actions_legal = True 
        except Exception: pass
            
        self.env = hisss.BattleSnakeGame(self.game_config)
        self.num_snakes = 4
        
        self.agent_ids = [f"snake_{i}" for i in range(self.num_snakes)]
        self._agent_ids = set(self.agent_ids)
        
        single_action_space = gym.spaces.Discrete(4)
        single_agent_obs_space = gym.spaces.Dict({
            "obs": gym.spaces.Box(low=0, high=255, shape=(21, 21, 22), dtype=np.uint8),
            "state": gym.spaces.Box(low=0, high=255, shape=(21, 21, 22), dtype=np.uint8)
        })
        
        self.action_space = gym.spaces.Dict({
            agent_id: single_action_space for agent_id in self.agent_ids
        })
        self.observation_space = gym.spaces.Dict({
            agent_id: single_agent_obs_space for agent_id in self.agent_ids
        })
        
        self.last_actions = {i: 0 for i in range(self.num_snakes)}
        self.turn_count = 0
        self.last_obs = None
        self.last_metrics = {}
        self.terminated_agents = set() 

    def _get_unpacked_obs(self):
        obs_dict, _, _ = self.env.get_obs()
        alive_ids = self.env.players_alive()
        
        unpacked = {}
        for i in range(self.num_snakes):
            if i in alive_ids:
                idx = alive_ids.index(i)
                unpacked[f"snake_{i}"] = {
                    "obs": obs_dict["actor_obs"][idx], 
                    "state": obs_dict["critic_obs"][idx]
                }
            else:
                unpacked[f"snake_{i}"] = {
                    "obs": np.zeros((21, 21, 22), dtype=np.uint8), 
                    "state": np.zeros((21, 21, 22), dtype=np.uint8)
                }
        return unpacked

    def _get_metrics(self, state_tensor):
        heads = np.argwhere(state_tensor[:, :, 1] == 1)
        head = tuple(heads[0]) if len(heads) > 0 else None
        
        foods = np.argwhere(state_tensor[:, :, 0] == 1)
        food_list = [tuple(f) for f in foods]
        
        length = int(np.sum(state_tensor[:, :, 2])) + 1
        return {"head": head, "food_list": food_list, "length": length}

    def _closest_food_dist(self, head, foods):
        if not head or not foods: return 999
        return min([abs(head[0] - f[0]) + abs(head[1] - f[1]) for f in foods])

    def reset(self, *, seed=None, options=None):
        self.env.reset()
        self.turn_count = 0
        self.last_actions = {i: 0 for i in range(self.num_snakes)}
        self.last_obs = self._get_unpacked_obs()
        
        self.last_metrics = {}
        for aid in self.agent_ids:
            self.last_metrics[aid] = self._get_metrics(self.last_obs[aid]["state"])
            
        self.terminated_agents = set()
        infos = {agent_id: {} for agent_id in self.agent_ids}
        return self.last_obs, infos

    def step(self, action_dict):
        self.turn_count += 1
        
        alive_ids = self.env.players_alive()
        actions = []
        penalty_flags = {i: False for i in range(self.num_snakes)}
        opposites = {0: 2, 2: 0, 1: 3, 3: 1}
        
        for i in alive_ids:
            action = int(action_dict.get(f"snake_{i}", 0))
            if self.turn_count > 1:
                last_act = self.last_actions[i]
                if action == opposites.get(last_act):
                    action = (last_act + 1) % 4  
                    penalty_flags[i] = True
                    
            actions.append(action)
            self.last_actions[i] = action

        prev_metrics = self.last_metrics

        try:
            raw_rewards, done, _ = self.env.step(tuple(actions))
        except ValueError as e:
            raw_rewards, done = [-2.0]*len(alive_ids), True

        current_alive = self.env.players_alive()
        game_over = bool(done) or len(current_alive) <= 1

        obs, curr_metrics, rewards, infos = {}, {}, {}, {}
        terminations = {"__all__": game_over}
        truncations = {"__all__": False}

        if not game_over:
            obs_unpacked = self._get_unpacked_obs()

        for i in range(self.num_snakes):
            agent_id = f"snake_{i}"
            
            if agent_id in self.terminated_agents:
                continue

            if game_over:
                obs[agent_id] = {
                    "obs": np.zeros((21, 21, 22), dtype=np.uint8), 
                    "state": np.zeros((21, 21, 22), dtype=np.uint8)
                }
                curr_metrics[agent_id] = {"head": None, "food_list": [], "length": 1}
            else:
                obs[agent_id] = obs_unpacked[agent_id]
                curr_metrics[agent_id] = self._get_metrics(obs[agent_id]["state"])

            is_dead = i not in current_alive
            terminations[agent_id] = game_over or is_dead
            truncations[agent_id] = False
            infos[agent_id] = {}

            # --- RULE 4: CLOSING THE LOOPHOLE ---
            raw = float(raw_rewards[alive_ids.index(i)]) if i in alive_ids else 0.0

            if is_dead or raw < 0:
                # 1. The Death State (Overrides everything)
                death_penalty = -10.0 + (self.turn_count * 0.2)
                rewards[agent_id] = min(-2.0, death_penalty)
            elif game_over:
                # 2. The Victory State (Last snake standing!)
                rewards[agent_id] = 10.0 + (self.turn_count * 0.1)
            else:
                # 3. The Survival State
                step_reward = -2.0 if penalty_flags[i] else 0.0
                rewards[agent_id] = raw + 0.01 + step_reward
                
                p_mets = prev_metrics.get(agent_id, {"head": None, "food_list": [], "length": 0})
                c_mets = curr_metrics.get(agent_id, {"head": None, "food_list": [], "length": 0})
                
                if c_mets["length"] > 3:
                    rewards[agent_id] += 0.01
                    
                prev_head = p_mets["head"]
                curr_head = c_mets["head"]
                prev_foods = p_mets["food_list"]
                curr_foods = c_mets["food_list"]
                
                if curr_head and prev_head:
                    if curr_head in prev_foods:
                        rewards[agent_id] += 1.0
                        
                    if prev_foods and curr_foods:
                        dist_before = self._closest_food_dist(prev_head, prev_foods)
                        dist_after = self._closest_food_dist(curr_head, curr_foods)
                        if dist_after < dist_before: rewards[agent_id] += 0.05
                        elif dist_after > dist_before: rewards[agent_id] -= 0.05
                            
                    enemy_heads = []
                    for layer_idx in range(self.num_snakes):
                        if layer_idx != i: 
                            e_id = f"snake_{layer_idx}"
                            if not game_over and e_id in curr_metrics:
                                e_head = curr_metrics[e_id]["head"]
                                if e_head: enemy_heads.append(e_head)
                    
                    if enemy_heads:
                        closest_e_dist_curr = min([abs(curr_head[0] - eh[0]) + abs(curr_head[1] - eh[1]) for eh in enemy_heads])
                        closest_e_dist_prev = min([abs(prev_head[0] - eh[0]) + abs(prev_head[1] - eh[1]) for eh in enemy_heads])
                        
                        if c_mets["length"] > 5 and closest_e_dist_curr < closest_e_dist_prev:
                            rewards[agent_id] += 0.05 
            
            if terminations[agent_id]:
                self.terminated_agents.add(agent_id)
        
        self.last_metrics = curr_metrics
        self.last_obs = obs
        return obs, rewards, terminations, truncations, infos
    
    def render(self):
        self.env.render()