from battlesnake_server import start_server
from rllib_agent import RLLibAgent

if __name__ == "__main__":
    agent = RLLibAgent()
    start_server(agent=agent, port=8000)
