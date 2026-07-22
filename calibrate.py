import hisss
import numpy as np

# Create a standard 15x15 game
cfg = hisss.standard_config()
cfg.w = 15
cfg.h = 15
env = hisss.BattleSnakeGame(cfg)

print("--- HISSS Spatial Calibration ---")
for action_int, name in [(0, "Action 0 (UP)"), (1, "Action 1 (RIGHT)"), (2, "Action 2 (DOWN)"), (3, "Action 3 (LEFT)")]:
    # Reset the environment safely
    env.reset()
    state = env.get_state()
    
    # Force a length-1 snake at exactly (7,7)
    # A length-1 snake has no neck, so it can't die from self-collision!
    state.snake_pos[0] = [(np.int32(7), np.int32(7))]
    state.snake_len[0] = np.int32(1)
    state.snake_health[0] = np.int32(100)
    env.set_state(state)
    
    # Force all snakes to take the action
    env.step((action_int, action_int, action_int, action_int))
    
    # Read the new coordinates
    new_head = env.get_state().snake_pos[0][0]
    dx = new_head[0] - 7
    dy = new_head[1] - 7
    
    print(f"{name} resulted in internal engine shift: X moves by {dx}, Y moves by {dy}")