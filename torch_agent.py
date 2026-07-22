import os
import random
import traceback
import numpy as np
import torch
import torch.nn as nn

from battlesnake_types import GameState, MoveAction, Direction, BaseAgent


class CNNStack(nn.Module):
    """Single CNN branch: 3 conv layers + adaptive avg pool to 2x2."""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(22, 16, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
        self.relu  = nn.ReLU()
        self.pool  = nn.AdaptiveAvgPool2d((2, 2))

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.pool(x)
        return x.flatten(1)  # (batch, 256)


class BattlesnakeNet(nn.Module):
    """
    Reconstructed RLLib ComplexInputNetwork + LSTM.

    8 CNN branches (4 _convs + 4 _value_branch_separate), each outputting
    256 features after AdaptiveAvgPool2d(2,2). Concatenated to 2048, fed
    into LSTM(2048->256), then action head Linear(256->4).
    """
    def __init__(self):
        super().__init__()
        # Policy branches (_convs)
        self.cnns_0_convs = CNNStack()
        self.cnns_1_convs = CNNStack()
        self.cnn_0_convs  = CNNStack()
        self.cnn_1_convs  = CNNStack()
        # Value branches (_value_branch_separate)
        self.cnns_0_vbs   = CNNStack()
        self.cnns_1_vbs   = CNNStack()
        self.cnn_0_vbs    = CNNStack()
        self.cnn_1_vbs    = CNNStack()

        self.lstm        = nn.LSTM(input_size=2048, hidden_size=256, batch_first=True)
        self.action_head = nn.Linear(256, 4)

    def forward(self, x, lstm_state):
        # 8 branches -> each (1, 256)
        branches = [
            self.cnns_0_convs(x),
            self.cnns_1_convs(x),
            self.cnn_0_convs(x),
            self.cnn_1_convs(x),
            self.cnns_0_vbs(x),
            self.cnns_1_vbs(x),
            self.cnn_0_vbs(x),
            self.cnn_1_vbs(x),
        ]
        combined = torch.cat(branches, dim=1)  # (1, 2048)
        lstm_in  = combined.unsqueeze(1)        # (1, 1, 2048)
        lstm_out, new_state = self.lstm(lstm_in, lstm_state)
        hidden  = lstm_out.squeeze(1)           # (1, 256)
        logits  = self.action_head(hidden)      # (1, 4)
        return logits, new_state


def load_weights(net, weights):
    def w(key):
        v = weights[key]
        return v if isinstance(v, torch.Tensor) else torch.tensor(v)

    # Policy branches
    for stack, prefix in [
        (net.cnns_0_convs, "cnns.0"),
        (net.cnns_1_convs, "cnns.1"),
        (net.cnn_0_convs,  "cnn_0"),
        (net.cnn_1_convs,  "cnn_1"),
    ]:
        stack.conv1.weight.data = w(f"{prefix}._convs.0._model.1.weight")
        stack.conv1.bias.data   = w(f"{prefix}._convs.0._model.1.bias")
        stack.conv2.weight.data = w(f"{prefix}._convs.1._model.1.weight")
        stack.conv2.bias.data   = w(f"{prefix}._convs.1._model.1.bias")
        stack.conv3.weight.data = w(f"{prefix}._convs.2._model.0.weight")
        stack.conv3.bias.data   = w(f"{prefix}._convs.2._model.0.bias")

    # Value branches (_value_branch_separate, first 3 conv layers only)
    for stack, prefix in [
        (net.cnns_0_vbs, "cnns.0"),
        (net.cnns_1_vbs, "cnns.1"),
        (net.cnn_0_vbs,  "cnn_0"),
        (net.cnn_1_vbs,  "cnn_1"),
    ]:
        stack.conv1.weight.data = w(f"{prefix}._value_branch_separate.0._model.1.weight")
        stack.conv1.bias.data   = w(f"{prefix}._value_branch_separate.0._model.1.bias")
        stack.conv2.weight.data = w(f"{prefix}._value_branch_separate.1._model.1.weight")
        stack.conv2.bias.data   = w(f"{prefix}._value_branch_separate.1._model.1.bias")
        stack.conv3.weight.data = w(f"{prefix}._value_branch_separate.2._model.0.weight")
        stack.conv3.bias.data   = w(f"{prefix}._value_branch_separate.2._model.0.bias")

    # LSTM
    net.lstm.weight_ih_l0.data = w("lstm.weight_ih_l0")
    net.lstm.weight_hh_l0.data = w("lstm.weight_hh_l0")
    net.lstm.bias_ih_l0.data   = w("lstm.bias_ih_l0")
    net.lstm.bias_hh_l0.data   = w("lstm.bias_hh_l0")

    # Action head
    net.action_head.weight.data = w("_logits_branch._model.0.weight")
    net.action_head.bias.data   = w("_logits_branch._model.0.bias")


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

        print(f"Loading weights from {weights_path} ...")
        weights = torch.load(weights_path, map_location="cpu", weights_only=True)
        self.net = BattlesnakeNet()
        load_weights(self.net, weights)
        self.net.eval()
        print("TorchAgent ready - Ray-free inference active.")
        self._lstm_states = {}

    def _fresh_lstm_state(self):
        return (
            torch.zeros(1, 1, 256),
            torch.zeros(1, 1, 256),
        )

    def get_name(self):   return "MAPPO Agent"
    def get_color(self):  return "#16D067"
    def get_author(self): return "Gluttony"

    def start(self, game_state: GameState):
        self._lstm_states[game_state.game.id] = self._fresh_lstm_state()

    def end(self, game_state: GameState):
        self._lstm_states.pop(game_state.game.id, None)

    def _encode_observation(self, game_state: GameState):
        grid = np.zeros((22, 21, 21), dtype=np.float32)

        for food in game_state.board.food:
            if 0 <= food.x < 21 and 0 <= food.y < 21:
                grid[0, food.y, food.x] = 1.0

        h = game_state.you.head
        if h and 0 <= h.x < 21 and 0 <= h.y < 21:
            grid[1, h.y, h.x] = 1.0

        for idx, pt in enumerate(game_state.you.body):
            if pt and 0 <= pt.x < 21 and 0 <= pt.y < 21:
                grid[min(2 + idx, 5), pt.y, pt.x] = 1.0

        opponents = [s for s in game_state.board.snakes if s.id != game_state.you.id]
        for opp in opponents:
            if opp.head and 0 <= opp.head.x < 21 and 0 <= opp.head.y < 21:
                grid[6, opp.head.y, opp.head.x] = 1.0
            for idx, pt in enumerate(opp.body):
                if pt and 0 <= pt.x < 21 and 0 <= pt.y < 21:
                    grid[min(7 + idx, 10), pt.y, pt.x] = 1.0

        return torch.tensor(grid).unsqueeze(0)  # (1, 22, 21, 21) ...

    def _safe_moves(self, game_state: GameState):
        head   = game_state.you.head
        neck   = game_state.you.body[1] if len(game_state.you.body) > 1 else None
        width  = game_state.board.width
        height = game_state.board.height
        safe   = []

        for d in Direction:
            nx, ny = head.x + d.dx, head.y + d.dy
            if nx < 0 or nx >= width or ny < 0 or ny >= height:
                continue
            if neck and nx == neck.x and ny == neck.y:
                if not (head.x == neck.x and head.y == neck.y):
                    continue
            safe.append(d)

        return safe

    def move(self, game_state: GameState) -> MoveAction:
        chosen = Direction.UP

        try:
            obs        = self._encode_observation(game_state)
            lstm_state = self._lstm_states.get(game_state.game.id, self._fresh_lstm_state())

            with torch.no_grad():
                logits, new_state = self.net(obs, lstm_state)

            self._lstm_states[game_state.game.id] = new_state
            action_idx = int(logits.argmax(dim=1).item())
            chosen     = self.ACTION_MAP.get(action_idx, Direction.UP)

        except Exception as e:
            print(f"Inference error - falling back: {e}")
            traceback.print_exc()

        safe = self._safe_moves(game_state)
        if chosen not in safe:
            print(f"Shield: blocked {chosen.value}, safe={[d.value for d in safe]}")
            chosen = random.choice(safe) if safe else Direction.UP

        return MoveAction(move=chosen)
