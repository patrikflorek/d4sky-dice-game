"""Module for evaluating and comparing different AI strategies in the Dice Game.

This module provides functionality to:
1. Run head-to-head matches between different AI implementations
2. Collect and analyze performance statistics
3. Generate detailed performance reports

The module supports three AI types:
- MinMax AI: Uses minimax algorithm with alpha-beta pruning
- RL AI: Uses reinforcement learning
- Optimal AI: Uses pre-computed optimal moves

Example:
    >>> evaluator = GameEvaluator(moves_dict)
    >>> evaluator.evaluate_ais(num_games=300)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, ClassVar, DefaultDict
from colorama import Fore, Style
import random
from collections import defaultdict
import os
from pathlib import Path

from .models import Position
from .min_max_ai import MinMaxAI
from .rl_ai import RLDiceAI
from .optimal_ai import OptimalAI

@dataclass
class MatchStats:
    """Statistics for matches between two AIs."""
    wins: int = 0
    total_games: int = 0
    
    @property
    def win_rate(self) -> float:
        """Calculate win rate as a percentage."""
        return (self.wins / self.total_games * 100) if self.total_games > 0 else 0

@dataclass
class GameEvaluator:
    """Evaluates performance of different AI implementations in the Dice Game."""
    
    # Game configuration
    moves: Dict[int, List[int]]
    show_games: bool = False
    rl_ai: Optional[RLDiceAI] = None
    
    # Constants
    DEFAULT_GAMES: ClassVar[int] = 300
    MODEL_FILENAME: ClassVar[str] = "rl_model.pt"
    
    # Internal state
    minmax_ai: MinMaxAI = field(init=False)
    optimal_ai: OptimalAI = field(init=False)
    matchup_stats: DefaultDict = field(default_factory=lambda: defaultdict(lambda: defaultdict(int)))
    
    def __post_init__(self):
        """Initialize AI instances and check for trained model."""
        self.minmax_ai = MinMaxAI(self.moves)
        self.optimal_ai = OptimalAI(self.moves)
        
        # Initialize or load RL AI
        if self.rl_ai is None:
            self.rl_ai = RLDiceAI(self.moves)
            self._check_rl_model()
    
    def _check_rl_model(self) -> None:
        """Check if trained RL model exists and warn if not found."""
        model_dir = Path(__file__).parent.parent / "models"
        model_path = model_dir / self.MODEL_FILENAME
        
        if not model_path.exists():
            print(f"{Fore.YELLOW}\nWarning: No trained RL model found at {model_path}{Style.RESET_ALL}")
            print(f"{Fore.YELLOW}Train the model first with: python main.py --evaluate --train --episodes 5000{Style.RESET_ALL}")

    def evaluate_ais(self, num_games: int = DEFAULT_GAMES) -> None:
        """Evaluate AI performance in round-robin matches.
        
        Args:
            num_games: Total number of games to play (divided among AI pairs)
        """
        if num_games < 3:
            raise ValueError("num_games must be at least 3 to evaluate all AI pairs")
            
        games_per_pair = num_games // 3  # Split games evenly between AI pairs
        
        print(f"{Fore.CYAN}\nEvaluating AI Performance...{Style.RESET_ALL}")
        print(f"Running {games_per_pair} games for each AI pair")
        
        # Run all matchups
        self._play_games("MinMax AI", "RL AI", self.minmax_ai, self.rl_ai, games_per_pair)
        self._play_games("MinMax AI", "Optimal AI", self.minmax_ai, self.optimal_ai, games_per_pair)
        self._play_games("RL AI", "Optimal AI", self.rl_ai, self.optimal_ai, games_per_pair)
        
        self._print_overall_stats()
        
    def _play_games(self, ai1_name: str, ai2_name: str, ai1, ai2, num_games: int) -> None:
        """Play a series of games between two AIs and record statistics.
        
        Args:
            ai1_name: Name of the first AI
            ai2_name: Name of the second AI
            ai1: First AI instance
            ai2: Second AI instance
            num_games: Number of games to play
        """
        print(f"\n{Fore.YELLOW}Playing {num_games} games: {ai1_name} vs {ai2_name}{Style.RESET_ALL}")
        
        for game in range(num_games):
            if self.show_games and game % 10 == 0:
                print(f"Game {game}/{num_games}")
                
            # Randomly decide who goes first
            if random.choice([True, False]):
                winner = self._play_single_game(ai1, ai2)
                self._update_stats(winner, ai1_name, ai2_name)
            else:
                winner = self._play_single_game(ai2, ai1)
                self._update_stats(winner, ai2_name, ai1_name)
    
    def _update_stats(self, winner: int, first_ai: str, second_ai: str) -> None:
        """Update matchup statistics based on game result.
        
        Args:
            winner: 1 if first_ai won, 2 if second_ai won
            first_ai: Name of the first AI
            second_ai: Name of the second AI
        """
        if winner == 1:
            self.matchup_stats[first_ai][second_ai] += 1
        else:
            self.matchup_stats[second_ai][first_ai] += 1
            
    def _play_single_game(self, first_ai, second_ai) -> int:
        """Play a single game between two AIs.
        
        Args:
            first_ai: First AI instance
            second_ai: Second AI instance
            
        Returns:
            1 if first_ai wins, 2 if second_ai wins
        """
        initial_face = random.choice([1, 2, 5, 6])
        position = Position(initial_face, initial_face)
        
        while True:
            if self.show_games:
                print(f"Position: face={position.face}, sum={position.suma}")
            
            # First AI's turn
            move = self._get_ai_move(first_ai, position)
            if self.show_games:
                print(f"First AI chose: {move}")
                
            position = position.make_move(move)
            
            if position.is_terminal():
                return 2  # Second AI wins (first AI went over)
            if position.suma == 21:
                return 1  # First AI wins
                
            # Second AI's turn
            move = self._get_ai_move(second_ai, position)
            if self.show_games:
                print(f"Second AI chose: {move}")
                
            position = position.make_move(move)
            
            if position.is_terminal():
                return 1  # First AI wins (second AI went over)
            if position.suma == 21:
                return 2  # Second AI wins
    
    def _get_ai_move(self, ai, position: Position) -> int:
        """Get move from an AI instance based on its type.
        
        Args:
            ai: AI instance
            position: Current game position
            
        Returns:
            Selected move
        """
        if isinstance(ai, MinMaxAI):
            _, move = ai.get_best_move(position)
            return move
        return ai.select_action(position)
                
    def _print_overall_stats(self) -> None:
        """Print detailed statistics for all AI matchups."""
        print(f"\n{Fore.GREEN}AI Head-to-Head Results:{Style.RESET_ALL}")
        print("=" * 50)
        
        ai_names = ["MinMax AI", "RL AI", "Optimal AI"]
        
        # Print header
        print("\n" + " " * 15, end="")
        for ai in ai_names:
            print(f"{ai:>15}", end="")
        print("\n" + "-" * 60)
        
        # Print matchup results
        for ai1 in ai_names:
            print(f"{ai1:<15}", end="")
            for ai2 in ai_names:
                if ai1 == ai2:
                    print(f"{'---':>15}", end="")
                else:
                    wins = self.matchup_stats[ai1][ai2]
                    total = wins + self.matchup_stats[ai2][ai1]
                    win_rate = (wins / total * 100) if total > 0 else 0
                    print(f"{f'{wins}/{total} ({win_rate:.1f}%)':>15}", end="")
            print()
            
        print("\n" + "=" * 50)
        print(f"\n{Fore.CYAN}Reading guide:{Style.RESET_ALL}")
        print("Each cell shows: wins/total (win%)")
        print("Row AI vs Column AI")
