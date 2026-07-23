import os
import threading
import ray
import numpy as np
from flask import Flask, request, jsonify
from ray.tune.registry import register_env

# Import your custom environment wrapper
from blackout_env import BattlesnakeBlackoutEnv

app = Flask(__name__)

# --- GLOBAL VARIABLES FOR RLLIB ---
shared_policy = None  
dummy_env = None

# Master dictionary that holds both the LSTM state AND the consistent enemy mapping
agent_memories = {}

# The Global Lock to prevent C++ Memory Race Conditions
env_lock = threading.Lock()

ACTION_MAP = {
    0: "up",
    1: "right",
    2: "down",
    3: "left"
}

def env_creator(env_config):
    return BattlesnakeBlackoutEnv(config=env_config)

def setup_rllib():
    global shared_policy, dummy_env
    
    base_dir = os.path.abspath("./battlesnake_checkpoint")
    if not os.path.exists(base_dir):
        raise ValueError(f"Checkpoint folder not found at {base_dir}! Please check the path.")

    valid_checkpoints = []
    for root, dirs, files in os.walk(base_dir):
        if any(f.startswith("algorithm_state") for f in files):
            valid_checkpoints.append(root)

    if not valid_checkpoints:
        raise ValueError(f"No valid RLlib model state found anywhere inside {base_dir}!")

    checkpoint_dir = max(valid_checkpoints, key=os.path.getmtime)
    print(f"--- Auto-detected latest checkpoint at: {checkpoint_dir} ---")

    print("--- Initializing Ray and Loading Model ---")
    ray.init(ignore_reinit_error=True)
    register_env("battlesnake_blackout_v0", env_creator)

    # 1. Initialize the Dummy C++ Environment
    dummy_env = BattlesnakeBlackoutEnv()
    dummy_env.reset() 

    from ray.rllib.policy.policy import Policy
    policy_dir = os.path.join(checkpoint_dir, "policies", "shared_policy")
    shared_policy = Policy.from_checkpoint(policy_dir)
    
    print("--- Neural Network Loaded Securely! ---")


def build_perfect_observation(game_state, memory_key):
    """
    Injects the Battlesnake JSON into the C++ Engine using a permanent 
    Enemy ID Map to prevent the LSTM memory from scrambling.
    """
    global dummy_env, agent_memories
    
    cpp_state = dummy_env.env.get_state()
    
    you = game_state["you"]
    my_id = you["id"]
    
    # 1. Map YOU to index 0
    cpp_state.snake_pos[0] = [(np.int32(pt["x"]), np.int32(pt["y"])) for pt in you["body"]]
    cpp_state.snake_health[0] = np.int32(you["health"])
    cpp_state.snake_len[0] = np.int32(you["length"])
    
    # 2. Extract the consistent enemy mapping we created on Turn 1
    enemy_map = agent_memories[memory_key]["enemy_map"]
    
    # Build a lookup for alive enemies by their permanently assigned index (1, 2, or 3)
    alive_opponents_by_index = {1: None, 2: None, 3: None}
    
    for opp in game_state["board"]["snakes"]:
        if opp["id"] == my_id:
            continue
            
        if opp["id"] not in enemy_map:
            available_indices = [idx for idx in [1, 2, 3] if idx not in enemy_map.values()]
            if available_indices:
                enemy_map[opp["id"]] = available_indices[0]
                
        if opp["id"] in enemy_map:
            idx = enemy_map[opp["id"]]
            alive_opponents_by_index[idx] = opp
            
    # 3. Inject Enemy snakes into the C++ Engine on their strict channels
    for i in range(1, 4):
        opp = alive_opponents_by_index[i]
        
        if opp is not None:
            cpp_state.snake_pos[i] = [(np.int32(pt["x"]), np.int32(pt["y"])) for pt in opp["body"]]
            cpp_state.snake_health[i] = np.int32(opp["health"])
            cpp_state.snake_len[i] = np.int32(opp["length"])
        else:
            survivors = [o for o in alive_opponents_by_index.values() if o is not None]
            if len(survivors) > 0:
                safe_body = [(np.int32(pt["x"]), np.int32(pt["y"])) for pt in survivors[0]["body"]]
            else:
                tail = you["body"][-1]
                safe_body = [(np.int32(tail["x"]), np.int32(tail["y"]))]
                
            cpp_state.snake_pos[i] = safe_body
            cpp_state.snake_health[i] = np.int32(0)
            cpp_state.snake_len[i] = np.int32(0)
            
    # 4. Map the Food
    food_coords = [(np.int32(f["x"]), np.int32(f["y"])) for f in game_state["board"]["food"]]
    if hasattr(cpp_state, 'food'):
        cpp_state.food = food_coords
    elif hasattr(cpp_state, 'food_pos'):
        cpp_state.food_pos = food_coords
        
    # 5. Inject Time/Starvation Data
    if hasattr(cpp_state, 'turn'):
        cpp_state.turn = np.int32(game_state["turn"])
    elif hasattr(cpp_state, 'step'):
        cpp_state.step = np.int32(game_state["turn"])
        
    # 6. Inject and Render!
    dummy_env.env.set_state(cpp_state)
    obs_dict = dummy_env._get_unpacked_obs()
    
    return obs_dict["snake_0"]


def get_safe_actions(game_state):
    """
    Parses the grid to find actions that do not result in instant death.
    """
    head = game_state["you"]["body"][0]
    width = game_state["board"]["width"]
    height = game_state["board"]["height"]
    
    # Map the 4 actions to the resulting (X, Y) coordinates
    potential_moves = {
        0: (head["x"], head["y"] + 1), # UP
        1: (head["x"] + 1, head["y"]), # RIGHT
        2: (head["x"], head["y"] - 1), # DOWN
        3: (head["x"] - 1, head["y"])  # LEFT
    }
    
    # Map out all physical hazards
    unsafe_coords = set()
    for snake in game_state["board"]["snakes"]:
        # We check all body pieces EXCEPT the very last tail piece,
        # because the tail will move forward out of the way on the next turn.
        body = snake["body"]
        for pt in body[:-1]:
            unsafe_coords.add((pt["x"], pt["y"]))
            
    safe_actions = []
    for action_int, (x, y) in potential_moves.items():
        # Check Wall Collisions
        if x < 0 or x >= width or y < 0 or y >= height:
            continue
        # Check Body Collisions
        if (x, y) in unsafe_coords:
            continue
            
        safe_actions.append(action_int)
        
    return safe_actions


# --- BATTLESNAKE API ROUTES ---

@app.get("/")
def on_info():
    return jsonify({
        "apiversion": "1",
        "author": "MAPPO_Agent",
        "color": "#8800FF", 
        "head": "all-seeing",
        "tail": "bolt"
    })

@app.post("/start")
def on_start():
    game_state = request.get_json()
    game_id = game_state["game"]["id"]
    snake_id = game_state["you"]["id"]
    memory_key = f"{game_id}_{snake_id}"
    
    global agent_memories
    enemy_map = {}
    opponents = [s for s in game_state["board"]["snakes"] if s["id"] != snake_id]
    for idx, opp in enumerate(opponents):
        enemy_map[opp["id"]] = idx + 1
        
    agent_memories[memory_key] = {
        "lstm_state": shared_policy.get_initial_state(),
        "enemy_map": enemy_map
    }
    
    return "ok"

@app.post("/move")
def on_move():
    game_state = request.get_json()
    game_id = game_state["game"]["id"]
    snake_id = game_state["you"]["id"]
    memory_key = f"{game_id}_{snake_id}"
    
    global agent_memories
    
    # Fallback in case the server restarted mid-game
    if memory_key not in agent_memories:
        enemy_map = {}
        opponents = [s for s in game_state["board"]["snakes"] if s["id"] != snake_id]
        for idx, opp in enumerate(opponents):
            enemy_map[opp["id"]] = idx + 1
            
        agent_memories[memory_key] = {
            "lstm_state": shared_policy.get_initial_state(),
            "enemy_map": enemy_map
        }
         
    with env_lock:
        my_obs = build_perfect_observation(game_state, memory_key)
        current_lstm_state = agent_memories[memory_key]["lstm_state"]
        
        action_int, new_lstm_state, info = shared_policy.compute_single_action(
            obs=my_obs,
            state=current_lstm_state,
            explore=False 
        )
        agent_memories[memory_key]["lstm_state"] = new_lstm_state

    # --- THE FULL ARMOR SHIELD ---
    safe_actions = get_safe_actions(game_state)
    
    # If the network's #1 choice results in immediate death (wall or body)...
    if action_int not in safe_actions:
        print(f"[SHIELD] Network attempted fatal move ({ACTION_MAP.get(action_int)}). Scanning alternatives...")
        
        logits = info.get("action_dist_inputs", info.get("behaviour_logits", None))
        chosen_safe_act = None
        
        if logits is not None:
            # Rank from highest confidence to lowest
            ranked_actions = np.argsort(logits)[::-1]
            
            for act in ranked_actions:
                if act in safe_actions:
                    chosen_safe_act = int(act)
                    print(f"[SHIELD] Smart Fallback successful! Executing: {ACTION_MAP[chosen_safe_act]}")
                    break
        
        # If we successfully found a secondary safe move, override.
        if chosen_safe_act is not None:
            action_int = chosen_safe_act
        else:
            # If everything failed or there are absolutely NO safe moves left (we are trapped)
            # Pick a random safe move if possible, otherwise accept death gracefully.
            if safe_actions:
                action_int = safe_actions[0]
            print(f"[SHIELD] Agent is trapped or no logits found. Final move locked: {ACTION_MAP.get(action_int)}")

    chosen_move_string = ACTION_MAP.get(action_int, "up")
    return jsonify({"move": chosen_move_string})

@app.post("/end")
def on_end():
    game_state = request.get_json()
    game_id = game_state["game"]["id"]
    snake_id = game_state["you"]["id"]
    memory_key = f"{game_id}_{snake_id}"
    
    global agent_memories
    if memory_key in agent_memories:
        del agent_memories[memory_key]
        
    return "ok"

if __name__ == "__main__":
    setup_rllib()
    port = int(os.environ.get("PORT", "8080"))
    print(f"Starting MAPPO Server on port {port}...")
    app.run(host="0.0.0.0", port=port)