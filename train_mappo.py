import os
import time
import random
# Force use of local AMD GPU if using ROCm
os.environ["HIP_VISIBLE_DEVICES"] = "0"

import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env
from blackout_env import BattlesnakeBlackoutEnv

CHECKPOINT_DIR = os.path.abspath("./battlesnake_checkpoint")

def env_creator(env_config):
    return BattlesnakeBlackoutEnv(config=env_config)

def policy_mapping_fn(agent_id, episode, worker, **kwargs):
    if agent_id == "snake_0":
        return "shared_policy"
    
    if worker is not None:
        if "frozen_opponent" in worker.policy_map:
            if random.random() < 0.5: 
                return "shared_policy"
            else: 
                return "frozen_opponent"
            
    return "shared_policy"

if __name__ == "__main__":
    print("--- Initializing Ray ---")
    print("--- Connecting to Ray Cluster ---")
    # address="auto" tells it to use the cluster instead of making a local one
    # runtime_env automatically zips your local files and sends them to the laptops!
    ray.init(
        address="auto", 
        ignore_reinit_error=True,
        runtime_env={
            "working_dir": ".",
            "excludes": [
                "*.whl",                   # Exclude massive installers
                "battlesnake_checkpoint/", # Exclude the checkpoints
                ".venv/",                  # Exclude the virtual environment
                "__pycache__"              # Exclude cached files
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
            policies={"shared_policy": (None, obs_space, act_space, {})},
            policy_mapping_fn=policy_mapping_fn,
            policies_to_train=["shared_policy"], 
        )
        .training(
            lr=5e-4, 
            lr_schedule=[
                [0, 5e-4],
                [10_000_000, 5e-5]
            ],
            gamma=0.99,
            train_batch_size=5120,     
            minibatch_size=320,
            num_epochs=5,
            model={
                # Layer 1: Shrinks 29x29 to 13x13
                # Layer 2: Shrinks 13x13 to 5x5
                # Layer 3: Shrinks 5x5 to 1x1 (Perfectly flattened!)
                "conv_filters": [[16, [5, 5], 2], [32, [5, 5], 2], [64, [5, 5], 1]],                
                "fcnet_hiddens": [256, 256],
                "use_lstm": True,
                "max_seq_len": 32,                  
                "lstm_cell_size": 256,      
            },
            entropy_coeff=0.05,             
        )
    )

    algo = config.build_algo()

    if os.path.exists(CHECKPOINT_DIR) and os.listdir(CHECKPOINT_DIR):
        print(f"\n[INFO] Found existing checkpoint! Resuming safely from: {CHECKPOINT_DIR}")
        algo.restore(CHECKPOINT_DIR)
        start_iter = algo.iteration + 1
    else:
        print("\n[INFO] No checkpoint found. Initializing a fresh network brain.")
        start_iter = 1

    print(f"\n--- Starting League Training Loop at Iteration {start_iter} ---")

    historical_weights = []
    
    for i in range(start_iter, 100000):
        start_time = time.time()
        result = algo.train()
        
        env_runners_stats = result.get("env_runners", result)
        
        policy_rewards_mean = env_runners_stats.get("policy_reward_mean", {})
        reward_mean = policy_rewards_mean.get("shared_policy", 0.0)
        
        policy_rewards_max = env_runners_stats.get("policy_reward_max", {})
        reward_max = policy_rewards_max.get("shared_policy", 0.0)
        
        len_mean = env_runners_stats.get("episode_len_mean", 0.0)
        episodes_completed = env_runners_stats.get("episodes_this_iter", 0)
        sps = result.get("num_env_steps_sampled_this_iter", 0) / (result.get("time_this_iter_s", 1) or 1)

        entropy_str = "N/A"
        if "info" in result and "learner" in result["info"]:
            learner_stats = result["info"]["learner"]
            if "shared_policy" in learner_stats:
                p_stats = learner_stats["shared_policy"].get("learner_stats", {})
                entropy = p_stats.get("entropy", None)
                if entropy is not None:
                    entropy_str = f"{entropy:.4f}"

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
            
            print(f"  --> [SAVING] Loaded random historical version into {policy_id} and saving checkpoint...")
            algo.save(checkpoint_dir=CHECKPOINT_DIR)

    ray.shutdown()