import json
import pytest

# Import the standalone Flask app from your new optimized server script
from rllib_agent import app

@pytest.fixture(scope="module")
def client():
    app.config['TESTING'] = True
    with app.test_client() as c:
        yield c

# Standard Battlesnake JSON request for testing
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
    """Tests if the root directory returns the snake's appearance."""
    r = client.get('/')
    assert r.status_code == 200
    data = json.loads(r.data)
    assert 'color' in data
    assert 'head' in data
    assert 'tail' in data

def test_move_returns_valid_direction(client):
    """Tests if the neural network and safety shield return a valid move."""
    # 1. Initialize the LSTM memory state
    client.post('/start', json=GAME_STATE)
    
    # 2. Request a move
    r = client.post('/move', json=GAME_STATE)
    assert r.status_code == 200
    
    # 3. Verify the move is valid
    data = json.loads(r.data)
    assert 'move' in data
    assert data['move'] in ['up', 'down', 'left', 'right']