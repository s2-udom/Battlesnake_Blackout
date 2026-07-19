import os
import threading
import ray
import numpy as np
from flask import Flask, request, jsonify
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env

# Import your custom environment wrapper
from blackout_env import BattlesnakeBlackoutEnv

app = Flask(__name__)

# --- GLOBAL VARIABLES FOR RLLIB ---
algo = None
dummy_env = None
lstm_memories = {}

# The Global Lock to prevent C++ Memory Race Conditions
env_lock = threading.Lock()

# Confirmed via Calibration Script
ACTION_MAP = {
    0: "up",
    1: "right",
    2: "down",
    3: "left"
}

def env_creator(env_config):
    return BattlesnakeBlackoutEnv(config=env_config)

def setup_rllib():
    global algo, dummy_env
    
    checkpoint_dir = os.path.abspath("./battlesnake_checkpoint")
    if not os.path.exists(checkpoint_dir):
        raise ValueError(f"Checkpoint not found at {checkpoint_dir}! Please check the path.")

    print("--- Initializing Ray and Loading Model ---")
    ray.init(ignore_reinit_error=True)
    register_env("battlesnake_blackout_v0", env_creator)

    # 1. Initialize the Dummy C++ Environment
    dummy_env = BattlesnakeBlackoutEnv()
    dummy_env.reset() 

    obs_space = dummy_env.observation_space["snake_0"]
    act_space = dummy_env.action_space["snake_0"]

    # 2. Rebuild the exact config used in training
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
    
    algo = config.build_algo()
    algo.restore(checkpoint_dir)
    print("--- Neural Network Loaded Securely! ---")


def build_perfect_observation(game_state):
    """
    Injects the Battlesnake JSON into the C++ Engine to perfectly recreate
    the Ego-Centric, 29x29, Blackout-masked tensor.
    """
    global dummy_env
    
    cpp_state = dummy_env.env.get_state()
    
    # 1. Map YOU to index 0
    you = game_state["you"]
    cpp_state.snake_pos[0] = [(np.int32(pt["x"]), np.int32(pt["y"])) for pt in you["body"]]
    cpp_state.snake_health[0] = np.int32(you["health"])
    cpp_state.snake_len[0] = np.int32(you["length"])
    
    # 2. Map Enemy snakes to indices 1, 2, and 3
    opponents = [s for s in game_state["board"]["snakes"] if s["id"] != you["id"]]
    for i in range(1, 4):
        if (i - 1) < len(opponents):
            opp = opponents[i - 1]
            cpp_state.snake_pos[i] = [(np.int32(pt["x"]), np.int32(pt["y"])) for pt in opp["body"]]
            cpp_state.snake_health[i] = np.int32(opp["health"])
            cpp_state.snake_len[i] = np.int32(opp["length"])
        else:
            # THE GHOST NECK FIX
            # Provide 2 coordinates so C++ deque size is never 0 or 1, preventing out_of_range aborts
            cpp_state.snake_pos[i] = [(np.int32(0), np.int32(0)), (np.int32(0), np.int32(0))]
            cpp_state.snake_health[i] = np.int32(0)
            cpp_state.snake_len[i] = np.int32(0)
            
    # 3. Map the Food
    food_coords = [(np.int32(f["x"]), np.int32(f["y"])) for f in game_state["board"]["food"]]
    if hasattr(cpp_state, 'food'):
        cpp_state.food = food_coords
    elif hasattr(cpp_state, 'food_pos'):
        cpp_state.food_pos = food_coords
        
    # 4. Inject Time/Starvation Data
    if hasattr(cpp_state, 'turn'):
        cpp_state.turn = np.int32(game_state["turn"])
    elif hasattr(cpp_state, 'step'):
        cpp_state.step = np.int32(game_state["turn"])
        
    # 5. Inject and Render!
    dummy_env.env.set_state(cpp_state)
    obs_dict = dummy_env._get_unpacked_obs()
    
    return obs_dict["snake_0"]


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
    
    global lstm_memories
    lstm_memories[memory_key] = algo.get_policy("shared_policy").get_initial_state()
    return "ok"

@app.post("/move")
def on_move():
    game_state = request.get_json()
    game_id = game_state["game"]["id"]
    snake_id = game_state["you"]["id"]
    memory_key = f"{game_id}_{snake_id}"
    
    if memory_key not in lstm_memories:
         lstm_memories[memory_key] = algo.get_policy("shared_policy").get_initial_state()
         
    # THE THREADING LOCK: Forces Flask's concurrent threads to wait their turn
    # This prevents the C++ memory from being wiped mid-calculation!
    with env_lock:
        my_obs = build_perfect_observation(game_state)
        current_lstm_state = lstm_memories[memory_key]
        
        action_int, new_lstm_state, _ = algo.compute_single_action(
            observation=my_obs,
            state=current_lstm_state,
            policy_id="shared_policy",
            explore=False 
        )
        
        lstm_memories[memory_key] = new_lstm_state

    # --- INFERENCE-SIDE BOWLING BUMPER ---
    my_body = game_state["you"]["body"]
    if len(my_body) > 1:
        head = my_body[0]
        neck = my_body[1]
        
        facing = None
        forbidden = None
        
        # Battlesnake coordinates: (0,0) is bottom-left
        if head["x"] > neck["x"]:
            facing = 1 # RIGHT
            forbidden = 3 # LEFT
        elif head["x"] < neck["x"]:
            facing = 3 # LEFT
            forbidden = 1 # RIGHT
        elif head["y"] > neck["y"]:
            facing = 0 # UP
            forbidden = 2 # DOWN
        elif head["y"] < neck["y"]:
            facing = 2 # DOWN
            forbidden = 0 # UP
            
        if facing is not None and action_int == forbidden:
            print(f"[SHIELD] Network attempted reverse move ({forbidden}). Forcing forward ({facing}).")
            action_int = facing
            
    chosen_move_string = ACTION_MAP.get(action_int, "up")
    return jsonify({"move": chosen_move_string})

@app.post("/end")
def on_end():
    game_state = request.get_json()
    game_id = game_state["game"]["id"]
    snake_id = game_state["you"]["id"]
    memory_key = f"{game_id}_{snake_id}"
    
    if memory_key in lstm_memories:
        del lstm_memories[memory_key]
        
    return "ok"

if __name__ == "__main__":
    setup_rllib()
    port = int(os.environ.get("PORT", "8080"))
    print(f"Starting MAPPO Server on port {port}...")
    app.run(host="0.0.0.0", port=port)