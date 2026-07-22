import hisss
env = hisss.BattleSnakeGame(hisss.standard_config())
print("Available methods:", [m for m in dir(env) if 'json' in m.lower() or 'state' in m.lower()])