import os
import time
import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env
from blackout_env import BattlesnakeBlackoutEnv

def env_creator(env_config):
    return BattlesnakeBlackoutEnv(config=env_config)

if __name__ == "__main__":
    CHECKPOINT_DIR = os.path.abspath("./battlesnake_checkpoint")
    
    if not os.path.exists(CHECKPOINT_DIR):
        print("ERROR: No saved model found! Run train_mappo.py first.")
        exit()

    ray.init(ignore_reinit_error=True)
    env_name = "battlesnake_blackout_v0"
    register_env(env_name, env_creator)

    # We need to define the exact same spaces and policy mapping as the training script
    dummy_env = BattlesnakeBlackoutEnv()
    obs_space = dummy_env.observation_space["snake_0"]
    act_space = dummy_env.action_space["snake_0"]

    def policy_mapping_fn(agent_id, episode, worker, **kwargs):
        return "shared_policy"

    # We only need 1 worker for watching the game, but we MUST include the model blueprint!
    config = (
        PPOConfig()
        .environment(env=env_name)
        .framework("torch")
        .api_stack(enable_rl_module_and_learner=False, enable_env_runner_and_connector_v2=False)
        .env_runners(num_env_runners=1, num_envs_per_env_runner=1)
        .resources(num_gpus=0)
        .multi_agent(
            policies={"shared_policy": (None, obs_space, act_space, {})},
            policy_mapping_fn=policy_mapping_fn,
        )
        .training(
            model={
                "conv_filters": [[32, [3, 3], 1], [64, [3, 3], 1], [128, [3, 3], 1]],
                "fcnet_hiddens": [256, 256],
            }
        )
    )

    print(f"Loading trained brain from: {CHECKPOINT_DIR}")
    algo = config.build()
    algo.restore(CHECKPOINT_DIR)

    # Initialize a local environment to watch
    env = BattlesnakeBlackoutEnv()
    obs, infos = env.reset()
    
    print("\n--- Let the Battle Begin! ---\n")
    env.render()
    
    done = False
    turn = 0
    
    while not done:
        turn += 1
        time.sleep(0.5) # Pause for half a second so you can actually see the moves
        
        # Ask the trained neural network for the best actions
        action_dict = {}
        for agent_id, agent_obs in obs.items():
            # compute_single_action automatically handles the neural network forward pass
            action_dict[agent_id] = algo.compute_single_action(
                observation=agent_obs,
                policy_id="shared_policy"
            )
            
        # Step the environment forward
        obs, rewards, terminations, truncations, infos = env.step(action_dict)
        
        # Print the board!
        print(f"\n--- Turn {turn} ---")
        env.render()
        
        if terminations.get("__all__", False) or truncations.get("__all__", False):
            done = True
            print("\n--- Game Over! ---")
            print(f"Final Rewards: {rewards}")

    algo.stop()
    ray.shutdown()