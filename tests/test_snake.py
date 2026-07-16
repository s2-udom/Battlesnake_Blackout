import json
import pytest
from hungry_agent import HungryAgent
from battlesnake_server import start_server
from battlesnake_types import Direction

def make_app():
    """Create a Flask test app with the HungryAgent."""
    from flask import Flask
    from unittest.mock import patch
    agent = HungryAgent()
    import battlesnake_server
    # Temporarily capture the app
    apps = []
    original_run = Flask.run
    def mock_run(self, *args, **kwargs):
        apps.append(self)
    with patch.object(Flask, 'run', mock_run):
        start_server(agent=agent, port=8000)
    return apps[0], agent

@pytest.fixture
def client():
    app, agent = make_app()
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_index(client):
    r = client.get('/')
    assert r.status_code == 200
    data = json.loads(r.data)
    assert 'color' in data

def test_move_returns_valid_direction(client):
    game_state = {
        "turn": 1,
        "game": {
            "id": "test-game",
            "source": "pytest",
            "timeout": 500,
            "ruleset": {
                "name": "standard",
                "version": "1.0",
                "settings": {
                    "foodSpawnChance": 15,
                    "hazardDamagePerTurn": 0,
                    "minimumFood": 1,
                    "viewRadius": 5
                }
            }
        },
        "board": {
            "height": 11, "width": 11,
            "food": [],
            "hazards": [],
            "snakes": [{
                "id": "you",
                "name": "test",
                "length": 3,
                "latency": "0",
                "health": 100,
                "head": {"x": 5, "y": 5},
                "body": [{"x": 5, "y": 5}, {"x": 5, "y": 4}, {"x": 5, "y": 3}],
                "customizations": {"color": "#ff0000"}
            }]
        },
        "you": {
            "id": "you",
            "name": "test",
            "length": 3,
            "latency": "0",
            "health": 100,
            "head": {"x": 5, "y": 5},
            "body": [{"x": 5, "y": 5}, {"x": 5, "y": 4}, {"x": 5, "y": 3}],
            "customizations": {"color": "#ff0000"}
        }
    }
    # Start the game first
    client.post('/start', json=game_state)
    r = client.post('/move', json=game_state)
    assert r.status_code == 200
    data = json.loads(r.data)
    assert data['move'] in ['up', 'down', 'left', 'right']
