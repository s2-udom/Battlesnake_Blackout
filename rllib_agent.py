import os
import random
import traceback
import numpy as np
import threading
import torch
import torch.nn as nn

from battlesnake_types import GameState, MoveAction, Direction, BaseAgent
from blackout_env import BattlesnakeBlackoutEnv 

class BattlesnakeNet(nn.Module):
    """
    The mathematically perfect PyTorch reconstruction of RLlib's VisionNetwork.
    Uses precise ZeroPad2d to match RLlib's internal 'same' padding logic.
    """
    def __init__(self, state_dict):
        super().__init__()
        
        # 1. Auto-detect the correct dictionary prefix from the PyTorch checkpoint
        self.prefix = ""
        for k in state_dict.keys():
            if k.endswith("_convs.0._model.1.weight"):
                self.prefix = k.replace("_convs.0._model.1.weight", "")
                break
                
        # 2. Reconstruct the precise RLlib Convolutional Math
        # Layer 1: [5x5] Kernel, Stride 2, Same Padding -> Output: 15x15
        self.conv1 = nn.Sequential(nn.ZeroPad2d(2), nn.Conv2d(22, 16, 5, stride=2), nn.ReLU())
        
        # Layer 2: [3x3] Kernel, Stride 2, Same Padding -> Output: 8x8
        self.conv2 = nn.Sequential(nn.ZeroPad2d(1), nn.Conv2d(16, 32, 3, stride=2), nn.ReLU())
        
        # Layer 3: [3x3] Kernel, Stride 1, Valid Padding -> Output: 6x6
        self.conv3 = nn.Sequential(nn.Conv2d(32, 64, 3, stride=1), nn.ReLU()) 
        
        self.flatten = nn.Flatten()
        
        # RLlib maps the flattened 64*6*6 (2304) tensor to a 256 feature vector
        self.linear = nn.Sequential(nn.Linear(64 * 6 * 6, 256), nn.ReLU())
        
        # 3. Reconstruct the LSTM and Action Head
        # Auto-detect LSTM input size in case RLlib concatenated 'obs' and 'state'
        self.lstm_in = state_dict["lstm.weight_ih_l0"].shape[1]
        self.lstm = nn.LSTM(input_size=self.lstm_in, hidden_size=256, batch_first=True)
        self.action_head = nn.Linear(256, 4)
        
        # Load the weights natively
        self._load_weights(state_dict)

    def _load_weights(self, state_dict):
        """Safely maps the RLlib dictionary keys to the native PyTorch layers."""
        def w(k):
            val = state_dict[k]
            return val.clone().detach() if isinstance(val, torch.Tensor) else torch.tensor(val)
            
        self.conv1[1].weight.data = w(self.prefix + "_convs.0._model.1.weight")
        self.conv1[1].bias.data   = w(self.prefix + "_convs.0._model.1.bias")
        
        self.conv2[1].weight.data = w(self.prefix + "_convs.1._model.1.weight")
        self.conv2[1].bias.data   = w(self.prefix + "_convs.1._model.1.bias")
        
        self.conv3[0].weight.data = w(self.prefix + "_convs.2._model.0.weight")
        self.conv3[0].bias.data   = w(self.prefix + "_convs.2._model.0.bias")
        
        self.linear[0].weight.data = w(self.prefix + "_convs.3._model.0.weight")
        self.linear[0].bias.data   = w(self.prefix + "_convs.3._model.0.bias")
        
        self.lstm.weight_ih_l0.data = w("lstm.weight_ih_l0")
        self.lstm.weight_hh_l0.data = w("lstm.weight_hh_l0")
        self.lstm.bias_ih_l0.data   = w("lstm.bias_ih_l0")
        self.lstm.bias_hh_l0.data   = w("lstm.bias_hh_l0")
        
        self.action_head.weight.data = w("_logits_branch._model.0.weight")
        self.action_head.bias.data   = w("_logits_branch._model.0.bias")

    def forward(self, x, lstm_state):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        x = self.linear(x) # Shape: [Batch, 256]
        
        # Safety catch: If ComplexInputNetwork appended a dummy state vector, pad it out.
        if self.lstm_in == 512:
            x = torch.cat([x, x], dim=1)
            
        x = x.unsqueeze(1) # Shape: [Batch, Sequence=1, Features]
        x, new_state = self.lstm(x, lstm_state)
        x = x.squeeze(1) # Back to [Batch, 256]
        
        logits = self.action_head(x)
        return logits, new_state


class TorchAgent(BaseAgent):

    ACTION_MAP = {
        0: Direction.UP,
        1: Direction.RIGHT,
        2: Direction.DOWN,
        3: Direction.LEFT,
    }

    def __init__(self):
        weights_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "battlesnake_checkpoint",
            "weights.pt",
        )
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"weights.pt not found at {weights_path}")

        print(f"Loading native PyTorch weights from {weights_path} ...")
        weights = torch.load(weights_path, map_location="cpu", weights_only=True)
        
        self.net = BattlesnakeNet(weights)
        self.net.eval()
        
        # Initialize the C++ vision engine
        self.dummy_env = BattlesnakeBlackoutEnv()
        self.dummy_env.reset()
        self.env_lock = threading.Lock()
        
        # Unified memory dictionary
        self._game_memories = {}
        
        print("TorchAgent ready - 0% Ray Overhead, 100% Native PyTorch Inference!")

    def _fresh_lstm_state(self):
        return (
            torch.zeros(1, 1, 256),
            torch.zeros(1, 1, 256),
        )

    def get_name(self):   return "MAPPO Predator"
    def get_color(self):  return "#16D067"
    def get_author(self): return "Gluttony"

    def start(self, game_state: GameState):
        key = f"{game_state.game.id}_{game_state.you.id}"
        
        enemy_map = {}
        opponents = [s for s in game_state.board.snakes if s.id != game_state.you.id]
        for idx, opp in enumerate(opponents):
            enemy_map[opp.id] = idx + 1
            
        self._game_memories[key] = {
            "lstm_state": self._fresh_lstm_state(),
            "enemy_map": enemy_map
        }

    def end(self, game_state: GameState):
        key = f"{game_state.game.id}_{game_state.you.id}"
        if key in self._game_memories:
            del self._game_memories[key]

    def _build_perfect_observation(self, game_state: GameState, memory_key: str):
        """Uses the C++ Engine to generate the exact 29x29 Ego-Centric tensor the network trained on."""
        with self.env_lock:
            cpp_state = self.dummy_env.env.get_state()
            my_id = game_state.you.id
            
            # Map YOU to index 0
            cpp_state.snake_pos[0] = [(np.int32(pt.x), np.int32(pt.y)) for pt in game_state.you.body]
            cpp_state.snake_health[0] = np.int32(game_state.you.health)
            cpp_state.snake_len[0] = np.int32(game_state.you.length)
            
            # Ensure safe fallback mapping if server restarted
            if memory_key not in self._game_memories:
                self.start(game_state)
                
            enemy_map = self._game_memories[memory_key]["enemy_map"]
            alive_opponents_by_index = {1: None, 2: None, 3: None}
            
            for opp in game_state.board.snakes:
                if opp.id == my_id:
                    continue
                if opp.id not in enemy_map:
                    available_indices = [idx for idx in [1, 2, 3] if idx not in enemy_map.values()]
                    if available_indices:
                        enemy_map[opp.id] = available_indices[0]
                if opp.id in enemy_map:
                    idx = enemy_map[opp.id]
                    alive_opponents_by_index[idx] = opp
                    
            for i in range(1, 4):
                opp = alive_opponents_by_index[i]
                if opp is not None:
                    cpp_state.snake_pos[i] = [(np.int32(pt.x), np.int32(pt.y)) for pt in opp.body]
                    cpp_state.snake_health[i] = np.int32(opp.health)
                    cpp_state.snake_len[i] = np.int32(opp.length)
                else:
                    survivors = [o for o in alive_opponents_by_index.values() if o is not None]
                    if len(survivors) > 0:
                        safe_body = [(np.int32(pt.x), np.int32(pt.y)) for pt in survivors[0].body]
                    else:
                        tail = game_state.you.body[-1]
                        safe_body = [(np.int32(tail.x), np.int32(tail.y))]
                    cpp_state.snake_pos[i] = safe_body
                    cpp_state.snake_health[i] = np.int32(0)
                    cpp_state.snake_len[i] = np.int32(0)
                    
            # Map Food & Time
            cpp_state.food = [(np.int32(f.x), np.int32(f.y)) for f in game_state.board.food]
            cpp_state.turn = np.int32(game_state.turn)
            
            self.dummy_env.env.set_state(cpp_state)
            obs_dict = self.dummy_env._get_unpacked_obs()
            
            # RLlib normalizes Box data dynamically. Re-create that normalization here:
            numpy_obs = obs_dict["snake_0"].astype(np.float32) / 255.0
            
            # PyTorch expects (Batch, Channels, Height, Width). We permute to shift the channels correctly.
            tensor_obs = torch.tensor(numpy_obs).permute(2, 0, 1).unsqueeze(0)
            
            return tensor_obs

    def _get_safe_actions(self, game_state: GameState):
        """The Full Armor Shield - Scans the board for all fatal collisions."""
        head = game_state.you.body[0]
        width = game_state.board.width
        height = game_state.board.height
        
        potential_moves = {
            0: (head.x, head.y + 1), # UP
            1: (head.x + 1, head.y), # RIGHT
            2: (head.x, head.y - 1), # DOWN
            3: (head.x - 1, head.y)  # LEFT
        }
        
        unsafe_coords = set()
        for snake in game_state.board.snakes:
            # We exclude the tail because it will move forward on the next turn
            for pt in snake.body[:-1]:
                unsafe_coords.add((pt.x, pt.y))
                
        safe_actions = []
        for action_int, (x, y) in potential_moves.items():
            if x < 0 or x >= width or y < 0 or y >= height:
                continue
            if (x, y) in unsafe_coords:
                continue
            safe_actions.append(action_int)
            
        return safe_actions

    def move(self, game_state: GameState) -> MoveAction:
        key = f"{game_state.game.id}_{game_state.you.id}"
        if key not in self._game_memories:
            self.start(game_state)
            
        chosen_dir = Direction.UP
        action_int = 0
        logits_array = None

        try:
            # Generate the true 29x29 ego-centric normalized observation
            obs = self._build_perfect_observation(game_state, key)
            lstm_state = self._game_memories[key]["lstm_state"]

            with torch.no_grad():
                logits, new_state = self.net(obs, lstm_state)
            
            self._game_memories[key]["lstm_state"] = new_state
            
            # Extract raw confidences for the Full Armor Shield
            logits_array = logits.squeeze(0).numpy()
            action_int = int(logits.argmax(dim=1).item())

        except Exception as e:
            print(f"Inference error - falling back: {e}")
            traceback.print_exc()

        # --- THE FULL ARMOR SHIELD ---
        safe_actions = self._get_safe_actions(game_state)
        
        if action_int not in safe_actions:
            print(f"[SHIELD] Network attempted fatal move ({self.ACTION_MAP.get(action_int)}). Scanning alternatives...")
            
            chosen_safe_act = None
            if logits_array is not None:
                # Rank from highest confidence to lowest
                ranked_actions = np.argsort(logits_array)[::-1]
                
                for act in ranked_actions:
                    if act in safe_actions:
                        chosen_safe_act = int(act)
                        print(f"[SHIELD] Smart Fallback successful! Executing: {self.ACTION_MAP[chosen_safe_act]}")
                        break
            
            if chosen_safe_act is not None:
                action_int = chosen_safe_act
            else:
                if safe_actions:
                    action_int = safe_actions[0]
                print(f"[SHIELD] Agent is trapped. Final move locked: {self.ACTION_MAP.get(action_int)}")

        chosen_dir = self.ACTION_MAP.get(action_int, Direction.UP)
        return MoveAction(move=chosen_dir)

if __name__ == "__main__":
    import sys
    from battlesnake_server import start_server

    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8080
    agent = TorchAgent()
    print(f"Booting up Native PyTorch Battlesnake Agent on port {port}...")
    start_server(agent=agent, port=port)