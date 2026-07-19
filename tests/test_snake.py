import json
import pytest
from flask import Flask
from unittest.mock import patch
from torch_agent import TorchAgent
from battlesnake_server import start_server

def make_app():
    agent = TorchAgent()
    apps = []
    original_run = Flask.run
    def mock_run(self, *args, **kwargs):
        apps.append(self)
    with patch.object(Flask, 'run', mock_run):
        start_server(agent=agent, port=8000)
    return apps[0], agent

@pytest.fixture(scope="module")
def client():
    app, agent = make_app()
    app.config['TESTING'] = True
    with app.test_client() as c:
        yield c

GAME_STATE = {
    "turn": 5,
    "game": {
        "id": "test-game-1",
        "source": "test",
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
        "food": [{"x": 3, "y": 3, "spawn_turn": 0}],
        "hazards": [],
        "snakes": [{
            "id": "you", "name": "test", "length": 3,
            "latency": "0", "health": 90,
            "head": {"x": 5, "y": 5},
            "body": [{"x": 5, "y": 5}, {"x": 5, "y": 4}, {"x": 5, "y": 3}],
            "customizations": {"color": "#16D067"}
        }]
    },
    "you": {
        "id": "you", "name": "test", "length": 3,
        "latency": "0", "health": 90,
        "head": {"x": 5, "y": 5},
        "body": [{"x": 5, "y": 5}, {"x": 5, "y": 4}, {"x": 5, "y": 3}],
        "customizations": {"color": "#16D067"}
    }
}

def test_index(client):
    r = client.get('/')
    assert r.status_code == 200
    data = json.loads(r.data)
    assert 'color' in data

def test_move_returns_valid_direction(client):
    client.post('/start', json=GAME_STATE)
    r = client.post('/move', json=GAME_STATE)
    assert r.status_code == 200
    data = json.loads(r.data)
    assert data['move'] in ['up', 'down', 'left', 'right']
