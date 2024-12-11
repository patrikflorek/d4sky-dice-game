"""Dice Game Implementation

This module implements the main game logic for the D4sky Dice Game, including:
- Game initialization and setup
- Turn management
- Player input handling
- Game state transitions
- Win/loss conditions
- AI opponent integration

The game supports three AI opponents:
1. MinMax AI: Uses minimax search with position evaluation
2. RL AI: Uses deep Q-learning with experience replay
3. Optimal AI: Uses expert system with strategic position tables

Example:
    >>> game = DiceGame(use_rl=True)  # Use RL AI opponent
    >>> game.train_rl_ai(episodes=1000)  # Optional: train RL AI
    >>> game.start_game()  # Start interactive game session
"""

import random
import time
from typing import Dict, List, Optional, ClassVar, Type, Union
from dataclasses import dataclass
from colorama import Fore, Style

from .models import Position
from .display import DiceDisplay
from .min_max_ai import MinMaxAI
from .rl_ai import RLDiceAI
from .optimal_ai import OptimalAI

# Type alias for AI opponents
AIOpponent = Union[MinMaxAI, RLDiceAI, OptimalAI]

@dataclass
class DiceGame:
    """Main game class handling game logic and player interaction.
    
    This class manages the game state, handles player input, and coordinates
    between the human player and AI opponent. It supports different AI types
    and provides a user-friendly interface with colored output.
    
    Attributes:
        moves (Dict[int, List[int]]): Valid moves for each face
        winning_numbers (List[int]): Face values good for starting
        display (DiceDisplay): Game state display handler
        ai (AIOpponent): AI opponent instance
        ai_name (str): Display name of AI opponent
        use_rl (bool): Whether using RL AI
        use_optimal (bool): Whether using Optimal AI
    """
    
    # Class constants
    TARGET_SUM: ClassVar[int] = 21
    MOVE_DELAY: ClassVar[float] = 0.5  # Seconds to wait between moves
    
    # Valid moves for each face (class constant)
    VALID_MOVES: ClassVar[Dict[int, List[int]]] = {
        1: [2, 3, 4, 5],  # 1 can move to 2,3,4,5
        2: [1, 3, 4, 6],  # 2 can move to 1,3,4,6
        3: [1, 2, 5, 6],  # 3 can move to 1,2,5,6
        4: [1, 2, 5, 6],  # 4 can move to 1,2,5,6
        5: [1, 3, 4, 6],  # 5 can move to 1,3,4,6
        6: [2, 3, 4, 5]   # 6 can move to 2,3,4,5
    }
    
    # Good starting numbers (class constant)
    STARTING_FACES: ClassVar[List[int]] = [1, 2, 5, 6]
    
    def __init__(self, use_rl: bool = False, use_optimal: bool = False):
        """Initialize game with specified AI opponent.
        
        Args:
            use_rl: If True, use RL AI opponent
            use_optimal: If True, use Optimal AI opponent (overrides use_rl)
        """
        self.moves = self.VALID_MOVES
        self.winning_numbers = self.STARTING_FACES
        self.display = DiceDisplay()
        self.use_rl = use_rl
        self.use_optimal = use_optimal
        
        # Initialize AI opponent
        if use_optimal:
            self.ai = OptimalAI(self.moves)
            self.ai_name = "Optimal AI"
        elif use_rl:
            self.ai = RLDiceAI(self.moves)
            self.ai_name = "RL AI"
        else:
            self.ai = MinMaxAI(self.moves)
            self.ai_name = "MinMax AI"
    
    def train_rl_ai(self, episodes: int = 1000) -> None:
        """Train the RL AI opponent if active.
        
        Args:
            episodes: Number of training episodes
            
        Raises:
            TypeError: If current AI is not RL-based
        """
        if not isinstance(self.ai, RLDiceAI):
            raise TypeError("Cannot train: AI is not RL-based!")
            
        print(Fore.CYAN + f"Training RL AI for {episodes} episodes..." + Style.RESET_ALL)
        self.ai.train(episodes)
        print(Fore.GREEN + "Training complete!" + Style.RESET_ALL)
    
    def _get_player_move(self, available_moves: List[int]) -> Optional[int]:
        """Get and validate player's move choice.
        
        Args:
            available_moves: List of valid moves
            
        Returns:
            Chosen move if valid, None if game should end
            
        Example:
            >>> move = game._get_player_move([2, 3, 4, 5])
            >>> if move is None:
            ...     print("Game ended")
            ... else:
            ...     print(f"Player chose {move}")
        """
        move_str = ",".join(map(str, available_moves))
        try:
            user_input = input(f"Your turn! Enter one of these numbers '{move_str}': ")
            if not user_input.isdigit():
                print(Fore.RED + "Please enter a valid number!" + Style.RESET_ALL)
                return None
                
            move = int(user_input)
            if move not in available_moves:
                print(Fore.RED + f"Invalid move! Please choose from: {move_str}" + Style.RESET_ALL)
                return None
                
            return move
            
        except KeyboardInterrupt:
            print("\nGame terminated by user.")
            return None
        except Exception as e:
            print(Fore.RED + f"An error occurred: {e}" + Style.RESET_ALL)
            return None
    
    def _get_ai_move(self, position: Position) -> int:
        """Get AI's chosen move.
        
        Args:
            position: Current game state
            
        Returns:
            AI's chosen move
        """
        print(f"\n{self.ai_name}'s turn...")
        time.sleep(self.MOVE_DELAY)
        
        if isinstance(self.ai, MinMaxAI):
            _, move = self.ai.get_best_move(position)
        else:
            move = self.ai.select_action(position)
            
        print(f"{self.ai_name} chose: {move}")
        time.sleep(self.MOVE_DELAY)
        return move
    
    def _check_game_end(self, position: Position, player_turn: bool) -> bool:
        """Check if game has ended and display result.
        
        Args:
            position: Current game state
            player_turn: True if it's player's turn
            
        Returns:
            True if game has ended, False otherwise
        """
        if position.is_terminal():
            self.display.display_position(position.face, position.suma)
            winner = "You" if player_turn else self.ai_name
            print(Fore.RED + f"Game Over! {winner} went over {self.TARGET_SUM}." + Style.RESET_ALL)
            return True
            
        if position.suma == self.TARGET_SUM:
            self.display.display_position(position.face, position.suma)
            winner = "You" if player_turn else self.ai_name
            print(Fore.GREEN + f"Game Over! {winner} won by reaching {self.TARGET_SUM}!" + Style.RESET_ALL)
            return True
            
        return False
    
    def start_game(self) -> None:
        """Start and manage a game session.
        
        This method:
        1. Initializes the game state
        2. Displays welcome message
        3. Randomly determines first player
        4. Manages turn-based gameplay
        5. Handles win/loss conditions
        
        The game continues until either:
        - A player reaches exactly 21 (win)
        - A player goes over 21 (loss)
        - The player quits (Ctrl+C)
        """
        # Initialize game state
        initial_face = random.choice(self.winning_numbers)
        position = Position(initial_face, initial_face)
        
        # Display welcome message
        print(Fore.YELLOW + "\nWelcome to the Dice Game!" + Style.RESET_ALL)
        print(Fore.CYAN + f"Try to get as close to {self.TARGET_SUM} as possible without going over.\n" + Style.RESET_ALL)
        
        self.display.display_position(position.face, position.suma)
        
        # Show winning chance for MinMax AI
        if isinstance(self.ai, MinMaxAI):
            result, _ = self.ai.get_best_move(position)
            if result < 0:
                print(Fore.GREEN + "I should win this game!\n" + Style.RESET_ALL)
            else:
                print(Fore.YELLOW + "If you play well, you can win!\n" + Style.RESET_ALL)

        # Randomly decide first player
        player_turn = random.choice([True, False])
        print(Fore.CYAN + f"{'You' if player_turn else self.ai_name} will go first!" + Style.RESET_ALL)
            
        # Main game loop
        while True:
            self.display.display_position(position.face, position.suma)
            available_moves = self.moves[position.face]
            
            # Handle player or AI turn
            if player_turn:
                move = self._get_player_move(available_moves)
                if move is None:
                    continue
            else:
                move = self._get_ai_move(position)
            
            # Update game state
            position = position.make_move(move)
            
            # Check for game end
            if self._check_game_end(position, player_turn):
                break
            
            # Switch turns
            player_turn = not player_turn
