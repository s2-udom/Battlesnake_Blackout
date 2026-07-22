import os
import time
import random
# Force use of local AMD GPU if using ROCm
os.environ["HIP_VISIBLE_DEVICES"] = "0"

import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env
from ray.rllib.policy.policy import Policy
from ray.rllib.algorithms.registry import POLICIES  # <--- Add this import

from blackout_env import BattlesnakeBlackoutEnv
from hungry_agent import HungryAgent
from battlesnake_types import GameState, Direction

CHECKPOINT_DIR = os.path.abspath("./battlesnake_checkpoint")

class HeuristicPolicy(Policy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.heuristic = HungryAgent()
        
        # A static ID so the HungryAgent's memory (AgentState) persists across the rollout
        self.dummy_game_id = "rl_training_game"
        
        if self.dummy_game_id not in self.heuristic.agent_states:
            self.heuristic.agent_states[self.dummy_game_id] = __import__("hungry_agent").AgentState(possible_food=[])
            
        self.action_map = {
            Direction.UP: 0, 
            Direction.RIGHT: 1, 
            Direction.DOWN: 2, 
            Direction.LEFT: 3
        }

    def compute_actions(self, obs_batch, state_batches=None, prev_action_batch=None, prev_reward_batch=None, info_batch=None, episodes=None, **kwargs):
        actions = []
        for obs in obs_batch:
            # RESTORED 22-CHANNEL LOGIC
            single_obs_dict = {
                "obs": obs[:18502].reshape((29, 29, 22)),
                "state": obs[18502:].reshape((29, 29, 22))
            }
            
            game_state = self._obs_to_game_state(single_obs_dict)
            
            try:
                move_action = self.heuristic.move(game_state)
                actions.append(self.action_map.get(move_action.move, 0))
            except Exception as e:
                actions.append(0) 
                
        return actions, state_batches or [], {}

    def _obs_to_game_state(self, obs_dict):
        grid = obs_dict["state"]
        
        food_list = []
        my_body = []
        my_head = {"x": 0, "y": 0} 
        enemy_bodies = []
        
        for x in range(15):
            for y in range(15):
                tx, ty = x + 7, y + 7
                
                if grid[tx, ty, 0] > 0:
                    food_list.append({"x": x, "y": y})
                if grid[tx, ty, 6] > 0:
                    my_head = {"x": x, "y": y}
                    my_body.append({"x": x, "y": y})
                elif grid[tx, ty, 4] > 0 or grid[tx, ty, 7] > 0:
                    my_body.append({"x": x, "y": y})
                elif grid[tx, ty, 13] > 0 or grid[tx, ty, 15] > 0 or grid[tx, ty, 16] > 0:
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
                "height": 15, "width": 15, "food": food_list, "hazards": [],
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

    def learn_on_batch(self, samples):
        return {} 

    def get_weights(self): return {}
    def set_weights(self, weights): pass

POLICIES["HeuristicPolicy"] = HeuristicPolicy


def env_creator(env_config):
    return BattlesnakeBlackoutEnv(config=env_config)

def policy_mapping_fn(agent_id, episode, worker, **kwargs):
    if agent_id == "snake_0":
        return "shared_policy"
    
    roll = random.random()
    if roll < 0.20:
        return "hungry_heuristic"
    elif roll < 0.60 and worker is not None and "frozen_opponent" in getattr(worker, "policy_map", {}):
        return "frozen_opponent"
    else:
        return "shared_policy"

if __name__ == "__main__":
    print("--- Initializing Ray ---")
    ray.init(
        address="auto", 
        ignore_reinit_error=True,
        runtime_env={
            "working_dir": ".",
            "excludes": [
                "*.whl", 
                "battlesnake_checkpoint/", 
                ".venv/", 
                "__pycache__"
            ]
        }
    )
    env_name = "battlesnake_blackout_v0"
    register_env(env_name, env_creator)

    dummy_env = BattlesnakeBlackoutEnv()
    obs_space = dummy_env.observation_space["snake_0"]
    act_space = dummy_env.action_space["snake_0"]

    config = (
        PPOConfig()
        .environment(env=env_name, env_config={})
        .framework("torch")
        .api_stack(enable_rl_module_and_learner=False, enable_env_runner_and_connector_v2=False)
        .env_runners(num_env_runners=5, num_envs_per_env_runner=1, sample_timeout_s=300)
        .resources(num_gpus=1)
        .multi_agent(
            policies={
                "shared_policy": (None, obs_space, act_space, {}),
                "hungry_heuristic": (HeuristicPolicy, obs_space, act_space, {}),
            },
            policy_mapping_fn=policy_mapping_fn,
            policies_to_train=["shared_policy"], 
        )
        .training(
            lr=5e-4, 
            lr_schedule=[[0, 5e-4], [10_000_000, 5e-5]],
            gamma=0.99,
            train_batch_size=5120,     
            minibatch_size=320,
            num_epochs=5,
            model={
                "conv_filters": [[16, [5, 5], 2], [32, [5, 5], 2], [64, [5, 5], 1]],                
                "fcnet_hiddens": [256, 256],
                "use_lstm": True,
                "max_seq_len": 32,                  
                "lstm_cell_size": 256,      
            },
            entropy_coeff_schedule=[[0, 0.05], [5_000_000, 0.01], [10_000_000, 0.001]]             
        )
    )

    algo = config.build_algo()

    latest_checkpoint = None
    if os.path.exists(CHECKPOINT_DIR):
        checkpoints = [
            os.path.join(CHECKPOINT_DIR, d) for d in os.listdir(CHECKPOINT_DIR) 
            if os.path.isdir(os.path.join(CHECKPOINT_DIR, d)) and d.startswith("iter_")
        ]
        if checkpoints:
            latest_checkpoint = max(checkpoints)

    if latest_checkpoint:
        print(f"\n[INFO] Resuming safely from: {latest_checkpoint}")
        algo.restore(latest_checkpoint)
        start_iter = algo.iteration + 1
    else:
        print("\n[INFO] No checkpoint found. Initializing a fresh network brain.")
        start_iter = 1

    print(f"\n--- Starting League Training Loop at Iteration {start_iter} ---")

    historical_weights = []
    
    for i in range(start_iter, 100000):
        result = algo.train()
        
        env_runners_stats = result.get("env_runners", result)
        reward_mean = env_runners_stats.get("policy_reward_mean", {}).get("shared_policy", 0.0)
        reward_max = env_runners_stats.get("policy_reward_max", {}).get("shared_policy", 0.0)
        len_mean = env_runners_stats.get("episode_len_mean", 0.0)
        episodes_completed = env_runners_stats.get("episodes_this_iter", 0)
        sps = result.get("num_env_steps_sampled_this_iter", 0) / (result.get("time_this_iter_s", 1) or 1)

        entropy_str = "N/A"
        if "info" in result and "learner" in result["info"]:
            learner_stats = result["info"]["learner"]
            if "shared_policy" in learner_stats:
                p_stats = learner_stats["shared_policy"].get("learner_stats", {})
                if "entropy" in p_stats:
                    entropy_str = f"{p_stats['entropy']:.4f}"

        print(
            f"Iter {i:04d} | "
            f"Reward: {reward_mean:>7.2f} (Max: {reward_max:>7.2f}) | "
            f"Len: {len_mean:>5.1f} | "
            f"Games: {episodes_completed:>3} | "
            f"Speed: {sps:>5.0f} steps/s | "
            f"Entropy: {entropy_str}"
        )

        if i % 5 == 0:
            policy_id = "frozen_opponent"

            if algo.get_policy(policy_id) is None:
                algo.add_policy(
                    policy_id=policy_id,
                    policy_cls=type(algo.get_policy("shared_policy")),
                    observation_space=obs_space,
                    action_space=act_space,
                    config={}
                )
            
            current_weights = algo.get_weights(["shared_policy"])["shared_policy"]
            historical_weights.append(current_weights)
            
            if len(historical_weights) > 20:
                historical_weights.pop(0)
            
            random_past_weight = random.choice(historical_weights)
            algo.set_weights({policy_id: random_past_weight})
            
            save_path = os.path.join(CHECKPOINT_DIR, f"iter_{i:06d}")
            algo.save(checkpoint_dir=save_path)
            print(f"  --> [SAVING] Loaded random historical version and saved to {save_path}")

    ray.shutdown()