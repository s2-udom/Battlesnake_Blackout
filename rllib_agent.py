import os
import traceback
import ray
import numpy as np
import random
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env
from battlesnake_types import GameState, MoveAction, Direction, BaseAgent
from blackout_env import BattlesnakeBlackoutEnv

def env_creator(env_config):
    return BattlesnakeBlackoutEnv(config=env_config)

class RLLibAgent(BaseAgent):
    def __init__(self):
        self.checkpoint_dir = os.path.abspath("./battlesnake_checkpoint")
        if not os.path.exists(self.checkpoint_dir):
            raise ValueError("No checkpoint found!")

        print("--- Initializing Ray ---")
        ray.init(ignore_reinit_error=True)
        register_env("battlesnake_blackout_v0", env_creator)

        dummy_env = BattlesnakeBlackoutEnv()
        obs_space = dummy_env.observation_space["snake_0"]
        act_space = dummy_env.action_space["snake_0"]

        config = (
            PPOConfig()
            .environment(env="battlesnake_blackout_v0")
            .framework("torch")
            .api_stack(enable_rl_module_and_learner=False, enable_env_runner_and_connector_v2=False)
            .env_runners(num_env_runners=1)
            .multi_agent(
                policies={"shared_policy": (None, obs_space, act_space, {})},
                policy_mapping_fn=lambda agent_id, *args, **kwargs: "shared_policy",
            )
            .training(
                model={
                "conv_filters": [[16, [5, 5], 2], [32, [5, 5], 2], [64, [5, 5], 1]],                
                    "fcnet_hiddens": [256, 256],
                    "use_lstm": True,
                    "max_seq_len": 32, 
                    "lstm_cell_size": 256,
                }
            )
        )
        self.algo = config.build()
        self.algo.restore(self.checkpoint_dir)
        
        self.action_map = {
            0: Direction.UP,
            1: Direction.RIGHT,
            2: Direction.DOWN,
            3: Direction.LEFT
        }
        
        self.active_games_memory = {}
        print("Brain loaded securely. All C++ engine vectors removed!")

    def get_name(self): return "MAPPO Agent"
    def get_color(self): return "#16D067"
    
    def start(self, game_state: GameState):
        game_id = game_state.game.id
        self.active_games_memory[game_id] = self.algo.get_policy("shared_policy").get_initial_state()

    def end(self, game_state: GameState):
        game_id = game_state.game.id
        if game_id in self.active_games_memory:
            del self.active_games_memory[game_id]

    def _manual_encode(self, game_state: GameState) -> dict:
        # Create the full 29x29 tensor as expected by the CNN
        grid = np.zeros((29, 29, 22), dtype=np.uint8)
        
        my_head = game_state.you.head
        CENTER = 14

        # ---------------------------------------------------------
        # 1. PURE CARTESIAN EGO-SHIFT
        # Matches hisss exactly: Head at (14,14), returning (X, Y)
        # ---------------------------------------------------------
        def get_ego_coords(game_x, game_y):
            dx = game_x - my_head.x
            dy = game_y - my_head.y
            return CENTER + dx, CENTER + dy

        # ---------------------------------------------------------
        # 2. FOG OF WAR FILTER
        # ---------------------------------------------------------
        def is_visible(x, y):
            return abs(x - my_head.x) <= 5 and abs(y - my_head.y) <= 5

        # Layer 1: Valid Board
        for y in range(game_state.board.height):
            for x in range(game_state.board.width):
                if is_visible(x, y):
                    ex, ey = get_ego_coords(x, y)
                    grid[ex, ey, 1] = 255 

        # Layer 0: Food
        for food in game_state.board.food:
            if is_visible(food.x, food.y):
                ex, ey = get_ego_coords(food.x, food.y)
                grid[ex, ey, 0] = 255
                
        # Layer 6: Your Head (Always locked dead center!)
        grid[CENTER, CENTER, 6] = 255
            
        # Layer 4: Your Body & Layer 7: Your Tail
        for idx, pt in enumerate(game_state.you.body):
            if is_visible(pt.x, pt.y):
                ex, ey = get_ego_coords(pt.x, pt.y)
                grid[ex, ey, 4] = 255 
                if idx == len(game_state.you.body) - 1:
                    grid[ex, ey, 7] = 255
                
        # Enemy elements
        opponents = [s for s in game_state.board.snakes if s.id != game_state.you.id]
        for opp in opponents:
            
            # Layer 15: Enemy Head
            if is_visible(opp.head.x, opp.head.y):
                ex, ey = get_ego_coords(opp.head.x, opp.head.y)
                grid[ex, ey, 15] = 255
                
            # Layer 13: Enemy Body & Layer 16: Enemy Tail
            for idx, pt in enumerate(opp.body):
                if is_visible(pt.x, pt.y):
                    ex, ey = get_ego_coords(pt.x, pt.y)
                    grid[ex, ey, 13] = 255 
                    if idx == len(opp.body) - 1:
                        grid[ex, ey, 16] = 255
                        
        return {"obs": grid, "state": grid}

    def move(self, game_state: GameState) -> MoveAction:
        chosen_move = Direction.UP
        game_id = game_state.game.id
        
        if game_id not in self.active_games_memory:
             self.active_games_memory[game_id] = self.algo.get_policy("shared_policy").get_initial_state()
        
        try:
            my_obs = self._manual_encode(game_state)
            current_state = self.active_games_memory[game_id]
            
            action_int, new_state, _ = self.algo.compute_single_action(
                observation=my_obs,
                state=current_state,
                policy_id="shared_policy",
                explore=False
            )
            
            self.active_games_memory[game_id] = new_state
            chosen_move = self.action_map.get(action_int, Direction.UP)

        except Exception as e:
            print(f"Inference fallback engaged: {e}")
            traceback.print_exc()

        # INVINCIBLE SAFETY SHIELD
        board_width = game_state.board.width
        board_height = game_state.board.height
        my_head = game_state.you.head
        my_neck = game_state.you.body[1] if len(game_state.you.body) > 1 else None

        safe_moves = []
        for d in [Direction.UP, Direction.RIGHT, Direction.DOWN, Direction.LEFT]:
            new_x = my_head.x + d.dx
            new_y = my_head.y + d.dy
            
            if new_x < 0 or new_x >= board_width or new_y < 0 or new_y >= board_height:
                continue
                
            if my_neck and new_x == my_neck.x and new_y == my_neck.y:
                if not (my_head.x == my_neck.x and my_head.y == my_neck.y):
                    continue
                    
            safe_moves.append(d)

        if chosen_move not in safe_moves:
            print(f"Shield activated! Blocked suicidal move: {chosen_move.value}")
            if safe_moves:
                chosen_move = random.choice(safe_moves)

        return MoveAction(move=chosen_move)

if __name__ == "__main__":
    import sys
    from battlesnake_server import start_server
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8080
    agent = RLLibAgent()
    start_server(agent, port)