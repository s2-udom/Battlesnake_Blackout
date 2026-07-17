from enum import Enum
from abc import abstractmethod
from pydantic import BaseModel, Field
from typing import List, Optional, Tuple, Any, Union

# ---------------------------------------------------------
# Misc Models
# ---------------------------------------------------------
class Point(BaseModel):
    x: int
    y: int

class Food(Point):
    # Field marked as Optional because the server might not send it every time
    spawn_turn: Optional[int] = None

class EliminatedCause(str, Enum):
    EliminatedByCollision = "snake-collision"
    EliminatedBySelfCollision = "snake-self-collision"
    EliminatedByOutOfHealth = "out-of-health"
    EliminatedByHeadToHeadCollision = "head-collision"
    EliminatedByOutOfBounds = "wall-collision"

class EliminationEvent(BaseModel):
    cause: EliminatedCause
    turn: int
    by: Optional[str] = None

# ---------------------------------------------------------
# Snake Model
# ---------------------------------------------------------
class SnakeCustomizations(BaseModel):
    # Changed from Tuple to Any/str to handle hex codes sent by the engine
    color: Any 
    head: Optional[str] = None
    tail: Optional[str] = None

class Snake(BaseModel):
    id: str
    name: str
    length: int
    latency: Optional[str] = None
    squad: Optional[str] = None
    health: Optional[int] = None
    head: Optional[Point] = None
    body: List[Optional[Point]] = []
    customizations: SnakeCustomizations
    elimination_event: Optional[EliminationEvent] = None

# ---------------------------------------------------------
# Game & Ruleset Models
# ---------------------------------------------------------
class RoyaleSettings(BaseModel):
    shrinkEveryNTurns: int

class SquadSettings(BaseModel):
    allowBodyCollisions: bool
    sharedElimination: bool
    sharedHealth: bool
    sharedLength: bool

class RulesetSettings(BaseModel):
    foodSpawnChance: int
    hazardDamagePerTurn: int
    minimumFood: int
    # Marked as Optional because standard games might not have a viewRadius
    viewRadius: Optional[int] = None 
    royale: Optional[RoyaleSettings] = None
    squad: Optional[SquadSettings] = None

class Ruleset(BaseModel):
    name: str
    version: str
    settings: RulesetSettings

class Game(BaseModel):
    id: str
    source: str
    timeout: int
    ruleset: Ruleset

# ---------------------------------------------------------
# Board & Root GameState Models
# ---------------------------------------------------------
class Board(BaseModel):
    height: int
    width: int
    food: List[Food]
    hazards: List[Point]
    snakes: List[Snake]

class GameState(BaseModel):
    turn: int
    game: Game
    board: Board
    you: Snake

# ---------------------------------------------------------
# Snake Action Model
# ---------------------------------------------------------
class Direction(str, Enum):
    UP = 'up'
    RIGHT = 'right'
    DOWN = 'down'
    LEFT = 'left'

    @property
    def board_delta(self) -> Tuple[int, int]:
        if self is Direction.DOWN: return 0, -1
        if self is Direction.UP: return 0, 1
        if self is Direction.LEFT: return -1, 0
        if self is Direction.RIGHT: return 1, 0
        return 0, 0
    
    @property
    def dx(self) -> int: return self.board_delta[0]
    @property
    def dy(self) -> int: return self.board_delta[1]

class MoveAction(BaseModel):
    move: Direction
@staticmethod
    def from_board_delta(delta: tuple) -> 'Direction':
        mapping = {
            (0, 1): Direction.UP,
            (0, -1): Direction.DOWN,
            (-1, 0): Direction.LEFT,
            (1, 0): Direction.RIGHT,
        }
        return mapping.get(delta, Direction.UP)
# ---------------------------------------------------------
# Base Agent Interface
# ---------------------------------------------------------
class BaseAgent:
    @abstractmethod
    def get_name(self): pass

    def get_color(self): return '#32CD32'
    def get_author(self): return None

    @abstractmethod
    def start(self, game_state: GameState): pass

    @abstractmethod
    def move(self, game_state: GameState) -> MoveAction: pass

    @abstractmethod
    def end(self, game_state: GameState): pass
