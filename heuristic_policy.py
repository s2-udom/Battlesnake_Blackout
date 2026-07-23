import numpy as np
from ray.rllib.policy.policy import Policy
from hungry_agent import HungryAgent, AgentState
from battlesnake_types import GameState, Direction

class HeuristicPolicy(Policy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.heuristic = HungryAgent()
        
        self.dummy_game_id = "rl_training_game"
        
        if self.dummy_game_id not in self.heuristic.agent_states:
            self.heuristic.agent_states[self.dummy_game_id] = AgentState(possible_food=[])
            
        self.action_map = {
            Direction.UP: 0, 
            Direction.RIGHT: 1, 
            Direction.DOWN: 2, 
            Direction.LEFT: 3
        }

    def compute_actions(self, obs_batch, state_batches=None, prev_action_batch=None, prev_reward_batch=None, info_batch=None, episodes=None, **kwargs):
        actions = []
        for obs in obs_batch:
            # Reshape straight to 11x11x22
            grid = obs.reshape((11, 11, 22))
            game_state = self._obs_to_game_state(grid)
            
            try:
                move_action = self.heuristic.move(game_state)
                actions.append(self.action_map.get(move_action.move, 0))
            except Exception as e:
                actions.append(0) 
                
        return actions, state_batches or [], {}

    def _obs_to_game_state(self, grid):
        food_list = []
        my_body = []
        # The head is now permanently at the center of the 11x11 crop
        my_head = {"x": 5, "y": 5} 
        enemy_bodies = []
        
        for x in range(11):
            for y in range(11):
                if grid[x, y, 0] > 0:
                    food_list.append({"x": x, "y": y})
                if grid[x, y, 6] > 0:
                    my_head = {"x": x, "y": y}
                    my_body.append({"x": x, "y": y})
                elif grid[x, y, 4] > 0 or grid[x, y, 7] > 0:
                    my_body.append({"x": x, "y": y})
                elif grid[x, y, 13] > 0 or grid[x, y, 15] > 0 or grid[x, y, 16] > 0:
                    enemy_bodies.append({"x": x, "y": y})

        state_dict = {
            "game": {
                "id": self.dummy_game_id, 
                "ruleset": {
                    "name": "standard", 
                    "version": "v1", 
                    "settings": {
                        "viewRadius": 5,
                        "foodSpawnChance": 15,          
                        "minimumFood": 1,               
                        "hazardDamagePerTurn": 14       
                    }
                },
                "map": "standard", "timeout": 500, "source": ""
            },
            "turn": 1,
            "board": {
                "height": 11, "width": 11, "food": food_list, "hazards": [],
                "snakes": [
                    {
                        "id": "heuristic_me", "name": "heuristic_me", "health": 100, 
                        "length": max(3, len(my_body)),
                        "head": my_head, "body": my_body if my_body else [my_head], 
                        "customizations": {"color": "#FFF", "head": "default", "tail": "default"}
                    },
                    {
                        "id": "enemy_blob", "name": "enemy_blob", "health": 100, 
                        "length": max(3, len(enemy_bodies)),
                        "head": enemy_bodies[0] if enemy_bodies else {"x": 0, "y": 0}, 
                        "body": enemy_bodies if enemy_bodies else [{"x": 0, "y": 0}], 
                        "customizations": {"color": "#FFF", "head": "default", "tail": "default"}
                    }
                ]
            },
            "you": {
                "id": "heuristic_me", "name": "heuristic_me", "health": 100, 
                "length": max(3, len(my_body)),
                "head": my_head, "body": my_body if my_body else [my_head], 
                "customizations": {"color": "#FFF", "head": "default", "tail": "default"}
            }
        }
        
        return GameState(**state_dict)

    def get_weights(self): return {}
    def set_weights(self, weights): pass