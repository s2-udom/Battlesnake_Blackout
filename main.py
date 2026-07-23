from battlesnake_server import start_server
from rllib_agent import TorchAgent

if __name__ == "__main__":
    agent = TorchAgent()
    start_server(agent=agent, port=8000)