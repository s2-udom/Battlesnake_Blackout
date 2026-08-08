import os
import time
import random
# Force use of local AMD GPU if using ROCm
os.environ["HIP_VISIBLE_DEVICES"] = "0"

import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env
from ray.rllib.algorithms.registry import POLICIES  

from blackout_env import BattlesnakeBlackoutEnv

CHECKPOINT_DIR = os.path.abspath("./battlesnake_checkpoint")

def create_heuristic_policy(observation_space, action_space, config):
    from heuristic_policy import HeuristicPolicy
    return HeuristicPolicy(observation_space, action_space, config)

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
                "battlesnake_checkpoint*", 
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
        .env_runners(num_env_runners=8, num_envs_per_env_runner=6, sample_timeout_s=300)
        .resources(num_gpus=1)
        .multi_agent(
            policies={
                "shared_policy": (None, obs_space, act_space, {}),
                "hungry_heuristic": (create_heuristic_policy, obs_space, act_space, {}), 
            },
            policy_mapping_fn=policy_mapping_fn,
            policies_to_train=["shared_policy"], 
        )
        .training(
            lr=5e-4, 
            lr_schedule=[[0, 5e-4], [10_000_000, 5e-5]],
            gamma=0.99,
            train_batch_size=49152,     
            minibatch_size=3072,
            num_epochs=5,
            model={
                "conv_filters": [[16, [3, 3], 1], [32, [3, 3], 1], [64, [3, 3], 1]],                
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
            print(f"  --> [SAVING] Saved League Checkpoint to {save_path}")

    ray.shutdown()