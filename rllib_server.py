import os
import threading
import torch
import torch.nn as nn
import numpy as np
from flask import Flask, request, jsonify

from blackout_env import BattlesnakeBlackoutEnv

# ---------------------------------------------------------
# 1. PyTorch Architecture (Reflecting true RLlib mixed padding)
# ---------------------------------------------------------
class BattlesnakeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            # Layer 1: Same padding (padding=1 keeps it 11x11)
            nn.Conv2d(22, 16, kernel_size=3, stride=1, padding=1), nn.ReLU(),
            # Layer 2: Same padding (padding=1 keeps it 11x11)
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1), nn.ReLU(),
            # Layer 3: Valid padding (padding=0 shrinks it to 9x9)
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=0), nn.ReLU(),
            nn.Flatten()
        )
        self.lstm = nn.LSTMCell(input_size=5184, hidden_size=256)
        self.logits = nn.Linear(256, 4)

    def forward(self, obs, h_in, c_in):
        x = obs.permute(0, 3, 1, 2)
        x = self.cnn(x)
        h_out, c_out = self.lstm(x, (h_in, c_in))
        action_logits = self.logits(h_out)
        return action_logits, h_out, c_out

# ---------------------------------------------------------
# 2. Server Setup & Engine Initialization
# ---------------------------------------------------------
app = Flask(__name__)

print("Loading Neural Network...")
model = BattlesnakeNet()
model.load_state_dict(torch.load("battlesnake_weights.pth", map_location=torch.device('cpu')), strict=True)
model.eval()

print("Initializing hisss Engine...")
dummy_env = BattlesnakeBlackoutEnv()
dummy_env.reset()
env_lock = threading.Lock()

agent_memories = {}
ACTION_MAP = {0: "up", 1: "right", 2: "down", 3: "left"}

# ---------------------------------------------------------
# 3. The Perfect Observation Builder
# ---------------------------------------------------------
def build_perfect_observation(game_state, memory_key):
    global dummy_env, agent_memories
    
    cpp_state = dummy_env.env.get_state()
    you = game_state["you"]
    my_id = you["id"]
    
    cpp_state.snake_pos[0] = [(np.int32(pt["x"]), np.int32(pt["y"])) for pt in you["body"]]
    cpp_state.snake_health[0] = np.int32(you["health"])
    cpp_state.snake_len[0] = np.int32(you["length"])
    
    enemy_map = agent_memories[memory_key]["enemy_map"]
    alive_opponents_by_index = {1: None, 2: None, 3: None}
    
    for opp in game_state["board"]["snakes"]:
        if opp["id"] == my_id: continue
        if opp["id"] not in enemy_map:
            available_indices = [idx for idx in [1, 2, 3] if idx not in enemy_map.values()]
            if available_indices: enemy_map[opp["id"]] = available_indices[0]
        if opp["id"] in enemy_map:
            alive_opponents_by_index[enemy_map[opp["id"]]] = opp
            
    for i in range(1, 4):
        opp = alive_opponents_by_index[i]
        if opp is not None:
            cpp_state.snake_pos[i] = [(np.int32(pt["x"]), np.int32(pt["y"])) for pt in opp["body"]]
            cpp_state.snake_health[i] = np.int32(opp["health"])
            cpp_state.snake_len[i] = np.int32(opp["length"])
        else:
            survivors = [o for o in alive_opponents_by_index.values() if o is not None]
            safe_body = [(np.int32(pt["x"]), np.int32(pt["y"])) for pt in survivors[0]["body"]] if survivors else [(np.int32(you["body"][-1]["x"]), np.int32(you["body"][-1]["y"]))]
            cpp_state.snake_pos[i] = safe_body
            cpp_state.snake_health[i] = np.int32(0)
            cpp_state.snake_len[i] = np.int32(0)
            
    cpp_state.food = [(np.int32(f["x"]), np.int32(f["y"])) for f in game_state["board"]["food"]]
    if hasattr(cpp_state, 'turn'): cpp_state.turn = np.int32(game_state["turn"])
    elif hasattr(cpp_state, 'step'): cpp_state.step = np.int32(game_state["turn"])
        
    dummy_env.env.set_state(cpp_state)
    obs_dict = dummy_env._get_unpacked_obs()
    
    obs_tensor = torch.tensor(obs_dict["snake_0"], dtype=torch.float32).unsqueeze(0)
    return obs_tensor

# ---------------------------------------------------------
# 4. Safe Actions Fallback
# ---------------------------------------------------------
def get_safe_actions(game_state):
    head = game_state["you"]["body"][0]
    width, height = game_state["board"]["width"], game_state["board"]["height"]
    potential_moves = {0: (head["x"], head["y"] + 1), 1: (head["x"] + 1, head["y"]), 2: (head["x"], head["y"] - 1), 3: (head["x"] - 1, head["y"])}
    
    unsafe_coords = {(pt["x"], pt["y"]) for snake in game_state["board"]["snakes"] for pt in snake["body"][:-1]}
    return [act for act, (x, y) in potential_moves.items() if 0 <= x < width and 0 <= y < height and (x, y) not in unsafe_coords]

# ---------------------------------------------------------
# 5. API Routes
# ---------------------------------------------------------
@app.get("/")
def on_info():
    return jsonify({"apiversion": "1", "author": "PyTorch_Agent", "color": "#8800FF", "head": "all-seeing", "tail": "bolt"})

@app.post("/start")
def on_start():
    game_state = request.get_json()
    game_id, snake_id = game_state["game"]["id"], game_state["you"]["id"]
    memory_key = f"{game_id}_{snake_id}"
    
    enemy_map = {opp["id"]: idx + 1 for idx, opp in enumerate(game_state["board"]["snakes"]) if opp["id"] != snake_id}
    agent_memories[memory_key] = {"h": torch.zeros(1, 256), "c": torch.zeros(1, 256), "enemy_map": enemy_map}
    return "ok"

@app.post("/move")
def on_move():
    game_state = request.get_json()
    memory_key = f"{game_state['game']['id']}_{game_state['you']['id']}"
    
    if memory_key not in agent_memories: on_start()
    mem = agent_memories[memory_key]
    
    with env_lock:
        with torch.no_grad():
            obs = build_perfect_observation(game_state, memory_key)
            logits, new_h, new_c = model(obs, mem["h"], mem["c"])
            agent_memories[memory_key]["h"], agent_memories[memory_key]["c"] = new_h, new_c
            action_int = torch.argmax(logits, dim=1).item()

    safe_actions = get_safe_actions(game_state)
    if action_int not in safe_actions:
        ranked_actions = torch.argsort(logits, descending=True).squeeze().tolist()
        chosen_safe_act = next((act for act in ranked_actions if act in safe_actions), None)
        action_int = chosen_safe_act if chosen_safe_act is not None else (safe_actions[0] if safe_actions else action_int)

    return jsonify({"move": ACTION_MAP.get(action_int, "up")})

@app.post("/end")
def on_end():
    game_state = request.get_json()
    memory_key = f"{game_state['game']['id']}_{game_state['you']['id']}"
    if memory_key in agent_memories: del agent_memories[memory_key]
    return "ok"

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8080"))
    print(f"Starting Battlesnake server on port {port}...")
    app.run(host="0.0.0.0", port=port)