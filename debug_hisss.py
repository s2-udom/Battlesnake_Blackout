import hisss
from blackout_env import BattlesnakeBlackoutEnv
import numpy as np

def debug_run():
    print("Initializing Environment...")
    env = BattlesnakeBlackoutEnv()
    
    print("Starting Reset loop (100 times)...")
    for i in range(100):
        print(f"Iteration {i}: Resetting...")
        env.reset()
        
        # Mocking an action dictionary
        actions = {"snake_0": 0, "snake_1": 0} 
        
        print(f"Iteration {i}: Stepping...")
        obs, rewards, done, trunc, info = env.step(actions)
        
        if done.get("__all__", False):
            print("Game Over detected in iteration.")
            
    print("SUCCESS: 100 resets/steps completed without crashing.")

if __name__ == "__main__":
    debug_run()