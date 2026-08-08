import os
import torch
import ray
from ray.rllib.policy.policy import Policy
from ray.tune.registry import register_env
from blackout_env import BattlesnakeBlackoutEnv

def env_creator(env_config):
    return BattlesnakeBlackoutEnv(config=env_config)

def extract_weights(output_file="battlesnake_weights.pth"):
    base_dir = os.path.abspath("./battlesnake_checkpoint")
    valid_checkpoints = [root for root, dirs, files in os.walk(base_dir) if any(f.startswith("algorithm_state") for f in files)]
    checkpoint_dir = max(valid_checkpoints, key=os.path.getmtime)
    policy_dir = os.path.abspath(os.path.join(checkpoint_dir, "policies", "shared_policy"))
    
    ray.init(ignore_reinit_error=True)
    register_env("battlesnake_blackout_v0", env_creator)
    policy = Policy.from_checkpoint(policy_dir)
    rllib_dict = policy.model.state_dict()

    key_mapping = {
        "_convs.0._model.1.weight": "cnn.0.weight",
        "_convs.0._model.1.bias": "cnn.0.bias",
        "_convs.1._model.1.weight": "cnn.2.weight",
        "_convs.1._model.1.bias": "cnn.2.bias",
        "_convs.2._model.0.weight": "cnn.4.weight",
        "_convs.2._model.0.bias": "cnn.4.bias",
        "lstm.weight_ih_l0": "lstm.weight_ih",
        "lstm.weight_hh_l0": "lstm.weight_hh",
        "lstm.bias_ih_l0": "lstm.bias_ih",
        "lstm.bias_hh_l0": "lstm.bias_hh",
        "_logits_branch._model.0.weight": "logits.weight",
        "_logits_branch._model.0.bias": "logits.bias"
    }

    clean_dict = {}
    for rllib_key, clean_key in key_mapping.items():
        if rllib_key not in rllib_dict:
            raise ValueError(f"CRITICAL: Could not find '{rllib_key}' in RLlib checkpoint.")
        clean_dict[clean_key] = rllib_dict[rllib_key]
        print(f"Successfully mapped '{rllib_key}' -> '{clean_key}'")

    torch.save(clean_dict, output_file)
    print(f"\nSuccess! Pure PyTorch weights saved to {os.path.abspath(output_file)}")
    ray.shutdown()

if __name__ == "__main__":
    extract_weights()