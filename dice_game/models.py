"""Game State Models for D4sky Dice Game

This module defines the core data structures and game state logic for the dice game.
The main component is the Position class which represents a game state and provides
methods for game progression and state evaluation.

Game Rules:
    - Players take turns rolling a die and adding the face value to their sum
    - The goal is to reach exactly 21 points
    - Going over 21 points results in a loss
    - Each face can only move to specific other faces (defined in moves dictionary)
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Optional, Set, FrozenSet, ClassVar
from functools import lru_cache

@dataclass(frozen=True)
class Position:
    """Represents a game state in the dice game.
    
    Attributes:
        suma (int): Current sum of the player's moves
        face (int): Current face showing on the die
        limit_suma (int): Target sum to win (default: 21)
    
    Example:
        >>> moves = {1: [2, 3, 4, 5], 2: [1, 3, 4, 6], ...}
        >>> pos = Position(suma=15, face=3)
        >>> possible_moves = pos.get_possible_moves(moves)
        >>> new_pos = pos.make_move(possible_moves[0])
    """
    
    suma: int
    face: int
    limit_suma: int = 21
    
    # Class constants for validation
    MIN_FACE: ClassVar[int] = 1
    MAX_FACE: ClassVar[int] = 6
    VALID_FACES: ClassVar[FrozenSet[int]] = frozenset(range(MIN_FACE, MAX_FACE + 1))
    
    def __post_init__(self) -> None:
        """Validate position attributes after initialization.
        
        Raises:
            ValueError: If any attributes have invalid values
        """
        if not isinstance(self.suma, int):
            raise ValueError("suma must be an integer")
        if not isinstance(self.face, int) or self.face not in self.VALID_FACES:
            raise ValueError(f"face must be an integer between {self.MIN_FACE} and {self.MAX_FACE}")
        if not isinstance(self.limit_suma, int) or self.limit_suma < 1:
            raise ValueError("limit_suma must be a positive integer")

    @lru_cache(maxsize=128)
    def get_possible_moves(self, moves: FrozenSet[tuple[int, tuple[int, ...]]]) -> List[int]:
        """Get list of valid moves from the current face.
        
        Args:
            moves: Frozen set of tuples mapping faces to possible next faces
                  Format: {(current_face, (possible_next_faces))}
        
        Returns:
            List of valid faces that can be reached from current face
            
        Raises:
            KeyError: If current face is not in moves dictionary
        """
        moves_dict = dict(moves)
        if self.face not in moves_dict:
            raise KeyError(f"Face {self.face} not found in moves dictionary")
        return list(moves_dict[self.face])
    
    def make_move(self, move: int) -> Position:
        """Create new position after making a move.
        
        Args:
            move: The face value to move to (will be added to suma)
            
        Returns:
            New Position with updated suma and face
            
        Example:
            >>> pos = Position(suma=15, face=3)
            >>> new_pos = pos.make_move(4)  # suma=19, face=4
            
        Raises:
            ValueError: If move is not a valid face value
        """
        if move not in self.VALID_FACES:
            raise ValueError(f"Invalid move: {move}. Must be between {self.MIN_FACE} and {self.MAX_FACE}")
        return Position(self.suma + move, move, self.limit_suma)
    
    def is_terminal(self) -> bool:
        """Check if position is terminal (game ending).
        
        Returns:
            True if suma > limit_suma (loss condition)
            False if game can continue
            
        Note:
            Winning condition (suma == limit_suma) is not considered terminal
            as it's handled separately in game logic.
        """
        return self.suma > self.limit_suma
    
    def is_winning(self) -> bool:
        """Check if position is a winning position.
        
        Returns:
            True if suma equals limit_suma (win condition)
            False otherwise
        """
        return self.suma == self.limit_suma
    
    def get_score(self) -> float:
        """Calculate a normalized score for the position.
        
        Returns:
            Float between -1 and 1:
                1.0: Winning position (suma == limit_suma)
                -1.0: Losing position (suma > limit_suma)
                Otherwise: Progress towards goal (suma/limit_suma)
        """
        if self.is_winning():
            return 1.0
        if self.is_terminal():
            return -1.0
        return self.suma / self.limit_suma
    
    def distance_to_win(self) -> int:
        """Calculate how many points needed to reach winning sum.
        
        Returns:
            Number of points needed to reach limit_suma.
            Returns negative value if already over limit.
        """
        return self.limit_suma - self.suma
    
    def is_safe_move(self, move: int) -> bool:
        """Check if making a move would keep suma within winning range.
        
        Args:
            move: The face value to check
            
        Returns:
            True if suma + move <= limit_suma
            False if move would exceed limit_suma
        """
        return self.suma + move <= self.limit_suma
    
    def __str__(self) -> str:
        """Return string representation of position.
        
        Returns:
            String in format "Position(suma=X, face=Y)"
        """
        return f"Position(suma={self.suma}, face={self.face})"
