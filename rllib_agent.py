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

        # We only need a dummy Python env to read the observation and action spaces
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
                    "conv_filters": [[32, [3, 3], 1], [64, [3, 3], 1], [128, [3, 3], 1]],
                    "fcnet_hiddens": [256, 256],
                    "use_lstm": True,
                    "lstm_cell_size": 256,
                }
            )
        )
        self.algo = config.build()
        self.algo.restore(self.checkpoint_dir)
        
        # Pure Python action mapping - No C++ imports needed!
        self.action_map = {
            0: Direction.UP,
            1: Direction.RIGHT,
            2: Direction.DOWN,
            3: Direction.LEFT
        }
        
        # Initialize memory state for the LSTM
        self.lstm_state = self.algo.get_policy("shared_policy").get_initial_state()
        print("Brain loaded securely. All C++ engine vectors removed!")

    def get_name(self): return "MAPPO Agent"
    def get_color(self): return "#16D067"
    
    def start(self, game_state: GameState):
        # Reset the LSTM short-term memory cleanly at the start of every game
        self.lstm_state = self.algo.get_policy("shared_policy").get_initial_state()

    def end(self, game_state: GameState):
        pass

    def _manual_encode(self, game_state: GameState) -> dict:
        """Natively builds the [21, 21, 22] observation tensor in Python.
        This completely eliminates C++ memory leakage overhead during live games.
        """
        grid = np.zeros((21, 21, 22), dtype=np.uint8)
        
        # Layer 0: Food positions
        for food in game_state.board.food:
            if 0 <= food.x < 21 and 0 <= food.y < 21:
                grid[food.y, food.x, 0] = 1
                
        # Layer 1: Your head
        my_head = game_state.you.head
        if 0 <= my_head.x < 21 and 0 <= my_head.y < 21:
            grid[my_head.y, my_head.x, 1] = 1
            
        # Layers 2-5: Your body segments
        for idx, pt in enumerate(game_state.you.body):
            if pt and 0 <= pt.x < 21 and 0 <= pt.y < 21:
                layer = min(2 + idx, 5)
                grid[pt.y, pt.x, layer] = 1
                
        # Layers 6-10: Enemy elements
# Layers 6-10: Enemy elements (UPDATED FOR 4-PLAYER)
        opponents = [s for s in game_state.board.snakes if s.id != game_state.you.id]
        for opp in opponents:
            if 0 <= opp.head.x < 21 and 0 <= opp.head.y < 21:
                grid[opp.head.y, opp.head.x, 6] = 1
            for idx, pt in enumerate(opp.body):
                if pt and 0 <= pt.x < 21 and 0 <= pt.y < 21:
                    layer = min(7 + idx, 10)
                    grid[pt.y, pt.x, layer] = 1
                    
        return {"obs": grid, "state": grid}

    def move(self, game_state: GameState) -> MoveAction:
        chosen_move = Direction.UP
        
        # ==========================================
        # 1. AI INFERENCE BLOCK (Pure Python/PyTorch)
        # ==========================================
        try:
            my_obs = self._manual_encode(game_state)
            
            action_int, self.lstm_state, _ = self.algo.compute_single_action(
                observation=my_obs,
                state=self.lstm_state,
                policy_id="shared_policy"
            )
            chosen_move = self.action_map.get(action_int, Direction.UP)

        except Exception as e:
            print(f"Inference fallback engaged: {e}")
            traceback.print_exc()

        # ==========================================
        # 2. INVINCIBLE SAFETY SHIELD
        # ==========================================
        board_width = game_state.board.width
        board_height = game_state.board.height
        my_head = game_state.you.head
        my_neck = game_state.you.body[1] if len(game_state.you.body) > 1 else None

        safe_moves = []
        for d in [Direction.UP, Direction.RIGHT, Direction.DOWN, Direction.LEFT]:
            new_x = my_head.x + d.dx
            new_y = my_head.y + d.dy
            
            # Rule 1: Do not hit walls
            if new_x < 0 or new_x >= board_width or new_y < 0 or new_y >= board_height:
                continue
                
            # Rule 2: Do not hit your own neck
            if my_neck and new_x == my_neck.x and new_y == my_neck.y:
                # Exception: On Turn 0, your neck and head are on the exact same square.
                if not (my_head.x == my_neck.x and my_head.y == my_neck.y):
                    continue
                    
            safe_moves.append(d)

        # If the AI chose a move that isn't safe, OVERRIDE IT!
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