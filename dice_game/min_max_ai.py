"""MinMax AI Implementation for D4sky Dice Game

This module implements a MinMax algorithm with position caching for the dice game.
The AI explores the game tree up to a fixed depth and uses a sophisticated position
evaluation function that considers:
1. Terminal states (wins/losses)
2. Strategic positions (power positions around 16-17)
3. Available moves (mobility)
4. Progress towards the goal

The implementation includes optimizations:
- Position caching to avoid re-exploring states
- Early pruning for terminal states
- Depth-aware scoring to prefer quicker wins
- Alpha-beta pruning for faster search
"""

from __future__ import annotations
from typing import Dict, Tuple, List, Optional, FrozenSet, ClassVar, NamedTuple
from functools import lru_cache
from dataclasses import dataclass
from .models import Position

class MoveScore(NamedTuple):
    """Represents a move and its evaluated score.
    
    Attributes:
        score: Evaluated position score
        move: Face value to move to
    """
    score: float
    move: int

@dataclass
class MinMaxAI:
    """MinMax algorithm implementation with position caching.
    
    The AI uses a depth-limited MinMax search with a sophisticated evaluation
    function. It caches positions to avoid re-computation and includes early
    pruning for terminal states.
    
    Attributes:
        MAX_DEPTH (int): Maximum search depth (class constant)
        moves (FrozenSet[tuple]): Valid moves for each face (immutable)
        position_cache (Dict): Cache of evaluated positions
        
    Example:
        >>> moves = {1: [2, 3, 4, 5], 2: [1, 3, 4, 6], ...}
        >>> ai = MinMaxAI(moves)
        >>> position = Position(suma=15, face=3)
        >>> score, best_move = ai.get_best_move(position)
    """
    
    # Class constants
    MAX_DEPTH: ClassVar[int] = 5
    INFINITY: ClassVar[float] = float('inf')
    
    # Strategic position scores
    BUILDING_PHASE_SCORE: ClassVar[float] = 0.5
    POWER_POSITION_SCORE: ClassVar[float] = 0.7
    DANGER_ZONE_SCORE: ClassVar[float] = 0.3
    CRITICAL_ZONE_SCORE: ClassVar[float] = 0.1
    MOBILITY_BONUS: ClassVar[float] = 0.1
    
    def __init__(self, moves: Dict[int, List[int]]):
        """Initialize MinMax AI with game rules.
        
        Args:
            moves: Dictionary mapping each face to allowed next faces
        """
        # Convert moves to immutable format for caching
        self.moves: FrozenSet[tuple[int, tuple[int, ...]]] = frozenset(
            (k, tuple(sorted(v))) for k, v in moves.items()
        )
        self.position_cache: Dict[tuple[int, int, bool], MoveScore] = {}
        
    @staticmethod
    @lru_cache(maxsize=1024)
    def _evaluate_position_cached(suma: int, face: int, depth: int, limit_suma: int = 21) -> float:
        """Static cached version of position evaluation.
        
        Args:
            suma: Current sum
            face: Current face
            depth: Search depth
            limit_suma: Target sum (default: 21)
            
        Returns:
            Evaluated score for the position
        """
        position = Position(suma, face, limit_suma)
        
        # Terminal positions with depth consideration
        if position.is_winning():
            return 1.0 + (1.0 / depth)  # Prefer winning sooner
            
        if position.is_terminal():
            return -1.0 - (1.0 / depth)  # Prefer losing later
            
        # Non-terminal position evaluation
        score = 0.0
        
        # Strategic scoring based on sum ranges
        if position.suma <= 15:
            # Building phase: reward progress
            score = MinMaxAI.BUILDING_PHASE_SCORE * (position.suma / position.limit_suma)
        elif position.suma <= 17:
            # Power position: highest non-terminal score
            score = MinMaxAI.POWER_POSITION_SCORE
        elif position.suma <= 19:
            # Danger zone: reduce score as we get closer to 21
            score = MinMaxAI.DANGER_ZONE_SCORE
        else:  # suma = 20
            # Very dangerous: prefer not to be here
            score = MinMaxAI.CRITICAL_ZONE_SCORE
            
        return score
    
    def evaluate_position(self, position: Position, depth: int) -> float:
        """Evaluate a position's strength using strategic heuristics.
        
        The evaluation considers multiple factors:
        1. Terminal states (win/loss)
        2. Strategic positions (building vs power positions)
        3. Available moves (mobility)
        4. Depth (prefer quicker wins, delayed losses)
        
        Args:
            position: Game state to evaluate
            depth: Current search depth (used for win/loss scoring)
            
        Returns:
            Float score between -1 and 1:
                > 0: Advantage for maximizing player
                < 0: Advantage for minimizing player
                ±1: Terminal states (win/loss)
        """
        # Get base score from cached evaluation
        score = self._evaluate_position_cached(
            position.suma,
            position.face,
            depth,
            position.limit_suma
        )
        
        # Add mobility bonus (not cached as it depends on moves)
        moves = position.get_possible_moves(self.moves)
        if len(moves) >= 3:
            score += self.MOBILITY_BONUS
            
        return score
        
    def get_best_move(
        self,
        position: Position,
        depth: int = 1,
        is_maximizing_player: bool = True,
        alpha: float = float('-inf'),
        beta: float = float('inf')
    ) -> MoveScore:
        """Find the best move using MinMax algorithm with alpha-beta pruning.
        
        Implements depth-limited MinMax search with position caching and
        alpha-beta pruning for efficient search. The search continues until either:
        1. A terminal state is reached
        2. Maximum depth is reached
        3. A cached position is found
        4. A branch is pruned by alpha-beta
        
        Args:
            position: Current game state
            depth: Current search depth (default: 1)
            is_maximizing_player: True if maximizing, False if minimizing
            alpha: Best score for maximizing player
            beta: Best score for minimizing player
            
        Returns:
            MoveScore containing position evaluation and best move
                
        Example:
            >>> score_move = ai.get_best_move(Position(15, 3))
            >>> print(f"Move to face {score_move.move} with score {score_move.score}")
        """
        # Check cache first for efficiency
        cache_key = (position.suma, position.face, is_maximizing_player)
        if cache_key in self.position_cache:
            return self.position_cache[cache_key]
            
        # Terminal position evaluation
        if position.is_terminal() or position.is_winning():
            result = MoveScore(self.evaluate_position(position, depth), position.face)
            self.position_cache[cache_key] = result
            return result
            
        # Depth limit reached
        if depth > self.MAX_DEPTH:
            result = MoveScore(self.evaluate_position(position, depth), position.face)
            self.position_cache[cache_key] = result
            return result
            
        # Initialize best score and move
        best_score = -self.INFINITY if is_maximizing_player else self.INFINITY
        best_move = position.face
        
        # Explore possible moves with alpha-beta pruning
        for move in position.get_possible_moves(self.moves):
            new_position = position.make_move(move)
            eval_score = self.get_best_move(
                new_position, depth + 1,
                not is_maximizing_player,
                alpha, beta
            ).score
            
            if is_maximizing_player:
                if eval_score > best_score:
                    best_score = eval_score
                    best_move = move
                alpha = max(alpha, best_score)
            else:
                if eval_score < best_score:
                    best_score = eval_score
                    best_move = move
                beta = min(beta, best_score)
                
            # Alpha-beta pruning
            if beta <= alpha:
                break
                
        result = MoveScore(best_score, best_move)
        self.position_cache[cache_key] = result
        return result
    
    def select_action(self, position: Position) -> int:
        """Interface compatibility with other AIs.
        
        Args:
            position: Current game state
            
        Returns:
            Best move (face value) for the position
        """
        return self.get_best_move(position).move
