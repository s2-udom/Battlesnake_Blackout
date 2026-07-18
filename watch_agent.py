import numpy as np
from blackout_env import BattlesnakeBlackoutEnv

print("--- Running Channel Diagnostic ---")
env = BattlesnakeBlackoutEnv()
obs, _ = env.reset()

# Grab the exact array being fed to Snake 0
ego_obs = obs["snake_0"]["obs"]

# In a 29x29 array, the Ego Head is permanently locked to the center (14, 14)
head_channels = np.where(ego_obs[14, 14] > 0)[0]

print(f"\n✅ YOUR EGO HEAD IS ON CHANNEL(S): {head_channels}")
print("If this number is NOT [6], your current manual encoder is blinding your snake!")