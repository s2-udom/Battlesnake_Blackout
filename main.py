from battlesnake_server import start_server
from hungry_agent import HungryAgent

if __name__ == "__main__":
    agent = HungryAgent()
    start_server(agent=agent, port=8000)
