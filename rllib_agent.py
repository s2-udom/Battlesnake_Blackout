import os
import sys
import numpy as np
from ray.rllib.policy.policy import Policy
from battlesnake_types import GameState, MoveAction, Direction, BaseAgent

class RLLibAgent(BaseAgent):
    def __init__(self):
        base_dir = os.path.abspath("./battlesnake_checkpoint")
        if not os.path.exists(base_dir):
            raise ValueError(f"No checkpoint base folder found at {base_dir}!")

        # 1. AUTO-DISCOVER LATEST CHECKPOINT ITERATION
        valid_checkpoints = []
        for root, dirs, files in os.walk(base_dir):
            if any(f.startswith("algorithm_state") for f in files):
                valid_checkpoints.append(root)

        if not valid_checkpoints:
            raise ValueError(f"No valid RLlib model state found anywhere inside {base_dir}!")

        # Grab the newest iteration folder (e.g., iter_010005)
        latest_checkpoint_dir = max(valid_checkpoints, key=os.path.getmtime)
        
        # Target the specific policy folder INSIDE the iteration folder
        policy_path = os.path.join(latest_checkpoint_dir, "policies", "shared_policy")
        
        print(f"--- Auto-detected latest policy at: {policy_path} ---")

        # 2. The Lightweight Load: Skips ray.init(), workers, and environments!
        self.policy = Policy.from_checkpoint(policy_path)

        # 3. Memory & Action Map Sync
        self.active_games_memory = {}
        
        # Action map mirroring train_mappo.py
        self.action_map = {
            0: Direction.UP,
            1: Direction.RIGHT,
            2: Direction.DOWN,
            3: Direction.LEFT
        }
        print("--- Pure Neural Brain Loaded Securely ---")

    def get_name(self):
        return "SCALES"

    def get_color(self):
        return "#8A2BE2"

    def get_author(self):
        return "Wrath"

    def start(self, game_state: GameState):
        """Initializes the LSTM memory state using the Policy API."""
        key = f"{game_state.game.id}_{game_state.you.id}"
        self.active_games_memory[key] = self.policy.get_initial_state()

    def move(self, game_state: GameState) -> MoveAction:
        """Processes the board and returns the model's chosen action."""
        key = f"{game_state.game.id}_{game_state.you.id}"
        
        # Fallback if server restarted but game is ongoing
        if key not in self.active_games_memory:
            self.active_games_memory[key] = self.policy.get_initial_state()
        
        lstm_state = self.active_games_memory[key]
        
        # Map the battlesnake JSON payload to a spatial dict
        obs_dict = self._manual_encode(game_state)

        # 4. Pure Inference Mode
        action, state_out, info = self.policy.compute_single_action(
            obs=obs_dict,
            state=lstm_state,
            explore=False 
        )
        
        # Save the new LSTM state for the next turn
        self.active_games_memory[key] = state_out

        # The Policy API sometimes returns numpy arrays instead of raw ints.
        # This safely extracts the integer value regardless of format.
        if isinstance(action, np.ndarray) or hasattr(action, "item"):
            safe_action = int(action.item())
        else:
            safe_action = int(action)
        
        return MoveAction(move=self.action_map[safe_action])

    def end(self, game_state: GameState):
        """Cleans up the LSTM state memory when the game finishes."""
        key = f"{game_state.game.id}_{game_state.you.id}"
        if key in self.active_games_memory:
            del self.active_games_memory[key]

    def _manual_encode(self, game_state: GameState) -> dict:
        """Translates the GameState object into the (29, 29, 22) Dict expected by CTDE."""
        grid = np.zeros((29, 29, 22), dtype=np.float32)
        my_head = game_state.you.head
        
        # The Ego-Centric Camera Shift (Anchors head to exact center: 14, 14)
        shift_x = 14 - my_head.x
        shift_y = 14 - my_head.y
        
        def set_obs(x, y, channel, val=1.0):
            nx = x + shift_x
            ny = y + shift_y
            if 0 <= nx < 29 and 0 <= ny < 29:
                grid[ny, nx, channel] = val

        # Layer 1: Valid Board
        for y in range(game_state.board.height):
            for x in range(game_state.board.width):
                set_obs(x, y, 1)

        # Layer 0: Food
        for food in game_state.board.food:
            set_obs(food.x, food.y, 0)

        # Layer 6: Your Head
        set_obs(my_head.x, my_head.y, 6)

        # Layer 4: Your Body & Layer 7: Your Tail
        for idx, pt in enumerate(game_state.you.body):
            set_obs(pt.x, pt.y, 4)
            if idx == len(game_state.you.body) - 1:
                set_obs(pt.x, pt.y, 7)

        # Enemy elements
        opponents = [s for s in game_state.board.snakes if s.id != game_state.you.id]
        for opp in opponents:
            # Layer 15: Enemy Head
            set_obs(opp.head.x, opp.head.y, 15)

            # Layer 13: Enemy Body & Layer 16: Enemy Tail
            for idx, pt in enumerate(opp.body):
                set_obs(pt.x, pt.y, 13)
                if idx == len(opp.body) - 1:
                    set_obs(pt.x, pt.y, 16)

        # Apply the 5-square Blackout logic directly to the actor's view
        masked_grid = np.zeros_like(grid)
        # Center is index 14. 5 squares out leaves valid data from indices 9 to 19.
        masked_grid[9:20, 9:20, :] = grid[9:20, 9:20, :]

        return {"obs": masked_grid, "state": grid}

if __name__ == "__main__":
    from battlesnake_server import start_server

    if len(sys.argv) < 2:
        port = 8080
    else:
        port = int(sys.argv[1])

    print(f"Booting up MAPPO Battlesnake Agent on port {port}...")
    agent = RLLibAgent()
    start_server(agent=agent, port=port)