"""Optimal AI Implementation for D4sky Dice Game

This module implements an expert system for the dice game that uses hand-crafted
strategies and position evaluation. It serves as both a strong opponent and a
source of expert moves for training the RL agent.

Strategic Concepts:
1. Winning Positions: Sums that allow forcing a win with specific faces
2. Safe Positions: Sums that minimize opponent's winning chances
3. Defensive Faces: Faces that offer better control (3,4 have balanced options)
4. Position Evaluation: Considers immediate wins, safety, and opponent responses

Example:
    >>> moves = {1: [2, 3, 4, 5], 2: [1, 3, 4, 6], ...}
    >>> ai = OptimalAI(moves)
    >>> position = Position(suma=15, face=3)
    >>> best_move = ai.select_action(position)
"""

from typing import Dict, List, Tuple, Optional, FrozenSet, ClassVar
from dataclasses import dataclass
from .models import Position

@dataclass
class OptimalAI:
    """Expert system implementation using strategic position evaluation.
    
    This AI uses hand-crafted strategies based on expert knowledge of the game.
    It maintains tables of winning and safe positions for each face, and uses
    a sophisticated scoring system that considers:
    1. Immediate wins/losses
    2. Strategic position values
    3. Defensive face preferences
    4. Opponent's potential responses
    
    Attributes:
        moves (FrozenSet): Valid moves for each face (immutable)
        winning_positions (Dict): Winning sum ranges for each face
        safe_positions (Dict): Safe sum values for each face
    """
    
    # Score constants for move evaluation
    WIN_SCORE: ClassVar[float] = 100.0
    LOSE_SCORE: ClassVar[float] = -100.0
    WINNING_POSITION_SCORE: ClassVar[float] = 50.0
    SAFE_POSITION_SCORE: ClassVar[float] = 30.0
    DEFENSIVE_FACE_BONUS: ClassVar[float] = 10.0
    DANGER_PENALTY_FACTOR: ClassVar[float] = 5.0
    OPPONENT_WIN_PENALTY: ClassVar[float] = 15.0
    
    # Strategic position constants
    POWER_POSITION_MIN: ClassVar[int] = 15  # Minimum sum for power positions
    DANGER_ZONE_START: ClassVar[int] = 18   # Start of dangerous territory
    DEFENSIVE_FACES: ClassVar[tuple[int, ...]] = (3, 4)  # Faces with balanced options
    
    def __init__(self, moves: Dict[int, List[int]]):
        """Initialize OptimalAI with game rules.
        
        Args:
            moves: Dictionary mapping each face to allowed next faces
                Format: {current_face: [possible_next_faces]}
        """
        # Convert moves to immutable format for caching
        self.moves: FrozenSet[tuple[int, tuple[int, ...]]] = frozenset(
            (k, tuple(sorted(v))) for k, v in moves.items()
        )
        
        # Define winning positions for each face
        # These ranges represent sums from which a win can be forced
        self.winning_positions = {
            1: range(16, 20),  # Can force win from 16-19
            2: range(15, 20),  # More options, can start from 15
            3: range(15, 19),  # Balanced range 15-18
            4: range(15, 19),  # Similar to 3, good control
            5: range(15, 19),  # Must be careful of opponent's 6
            6: range(15, 18),  # Limited high range due to risk
        }
        
        # Define safe positions for each face
        # These are positions that minimize opponent's winning chances
        self.safe_positions = {
            1: [16, 17],  # Low face needs higher sums
            2: [16, 17],  # Similar to 1, good defensive
            3: [15],      # Mid face can control from 15
            4: [15],      # Similar to 3, balanced
            5: [15],      # Must stay low due to high value
            6: [15],      # Must stay low due to highest value
        }
        
    def _can_win_directly(self, position: Position) -> Optional[int]:
        """Check if we can win in one move.
        
        Args:
            position: Current game state
            
        Returns:
            Winning move if available, None otherwise
            
        Example:
            >>> pos = Position(suma=18, face=3)
            >>> ai._can_win_directly(pos)  # Returns 3 if 3 is a valid move
        """
        for move in position.get_possible_moves(self.moves):
            if position.suma + move == position.limit_suma:
                return move
        return None
    
    def _is_winning_position(self, suma: int, face: int) -> bool:
        """Check if a given sum-face combination is a winning position.
        
        A winning position is one from which a win can be forced with
        proper play, regardless of opponent's moves.
        
        Args:
            suma: Current sum
            face: Current face
            
        Returns:
            True if position is in winning range for face
            
        Example:
            >>> ai._is_winning_position(16, 1)  # True
            >>> ai._is_winning_position(14, 1)  # False
        """
        return suma in self.winning_positions[face]
    
    def _is_safe_position(self, suma: int, face: int) -> bool:
        """Check if a given sum-face combination is a safe position.
        
        A safe position is one that minimizes the opponent's winning
        chances while maintaining good winning chances for us.
        
        Args:
            suma: Current sum
            face: Current face
            
        Returns:
            True if position is in safe range for face
            
        Example:
            >>> ai._is_safe_position(15, 3)  # True
            >>> ai._is_safe_position(18, 3)  # False
        """
        return suma in self.safe_positions[face]
    
    def _evaluate_move(self, position: Position, move: int) -> float:
        """Evaluate a move based on strategic principles.
        
        The evaluation considers multiple factors with different weights:
        1. Immediate wins/losses (±100.0)
        2. Winning positions (+50.0)
        3. Safe positions (+30.0)
        4. Defensive faces (+10.0)
        5. Danger zone penalties (-5.0 per point over 18)
        6. Opponent winning moves (-15.0 per winning move)
        
        Args:
            position: Current game state
            move: Proposed move
            
        Returns:
            Score where higher is better
            
        Example:
            >>> pos = Position(suma=15, face=3)
            >>> ai._evaluate_move(pos, 4)  # High score for good move
            >>> ai._evaluate_move(pos, 6)  # Low score for risky move
        """
        new_suma = position.suma + move
        
        # Immediate win/loss evaluation
        if new_suma == position.limit_suma:
            return self.WIN_SCORE
        if new_suma > position.limit_suma:
            return self.LOSE_SCORE
            
        score = 0.0
        
        # Strategic position evaluation
        if self._is_winning_position(new_suma, move):
            score += self.WINNING_POSITION_SCORE
        elif self._is_safe_position(new_suma, move):
            score += self.SAFE_POSITION_SCORE
            
        # Defensive face bonus
        if move in self.DEFENSIVE_FACES:
            score += self.DEFENSIVE_FACE_BONUS
            
        # Danger zone penalty
        if new_suma > self.DANGER_ZONE_START and new_suma != position.limit_suma:
            score -= (new_suma - self.DANGER_ZONE_START) * self.DANGER_PENALTY_FACTOR
            
        # Opponent winning moves penalty
        opponent_winning_moves = 0
        moves_dict = dict(self.moves)
        for opponent_move in moves_dict[move]:
            if new_suma + opponent_move == position.limit_suma:
                opponent_winning_moves += 1
        score -= opponent_winning_moves * self.OPPONENT_WIN_PENALTY
        
        return score
    
    def get_best_move(self, position: Position) -> Tuple[float, int]:
        """Get the best move based on optimal strategy.
        
        The selection process follows this priority:
        1. Take immediate wins if available
        2. Otherwise, evaluate all moves considering:
           - Strategic position value
           - Safety of resulting position
           - Opponent's potential responses
        
        Args:
            position: Current game state
            
        Returns:
            Tuple of (evaluation_score, chosen_move)
            
        Example:
            >>> pos = Position(suma=15, face=3)
            >>> score, move = ai.get_best_move(pos)
            >>> print(f"Move {move} with score {score}")
        """
        # First, check for immediate win
        direct_win = self._can_win_directly(position)
        if direct_win is not None:
            return (self.WIN_SCORE, direct_win)
            
        # Evaluate all possible moves
        move_scores = {}
        for move in position.get_possible_moves(self.moves):
            move_scores[move] = self._evaluate_move(position, move)
            
        # Choose the move with highest score
        best_move = max(move_scores.items(), key=lambda x: x[1])
        return (best_move[1], best_move[0])
    
    def select_action(self, position: Position) -> int:
        """Choose the best move for the current position.
        
        This is the main interface method used by the game engine.
        It returns only the move, discarding the evaluation score.
        
        Args:
            position: Current game state
            
        Returns:
            Best move (face value) for the position
            
        Example:
            >>> pos = Position(suma=15, face=3)
            >>> move = ai.select_action(pos)
            >>> print(f"Choosing move: {move}")
        """
        _, move = self.get_best_move(position)
        return move
