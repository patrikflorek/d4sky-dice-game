"""D4sky Dice Game - Main Entry Point

This module provides the command-line interface for the dice game, allowing users to:
1. Play against different AI opponents (MinMax, Optimal, or RL-based)
2. Train the RL AI with custom parameters
3. Evaluate AI performance in head-to-head matches

Example usage:
    # Play against MinMax AI (default)
    python main.py
    
    # Train and play against RL AI
    python main.py --rl --train --episodes 2000
    
    # Evaluate AI performance
    python main.py --evaluate --num-games 500
"""

import argparse
from typing import Optional
from colorama import Fore, Style

from dice_game.evaluator import GameEvaluator
from dice_game.game import DiceGame
from dice_game.rl_ai import RLDiceAI
from dice_game.optimal_ai import OptimalAI
from dice_game.min_max_ai import MinMaxAI

def setup_argument_parser() -> argparse.ArgumentParser:
    """Create and configure the command-line argument parser."""
    parser = argparse.ArgumentParser(
        description='D4sky Dice Game - Play against or evaluate different AI opponents',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # AI selection arguments
    ai_group = parser.add_argument_group('AI Selection')
    ai_group.add_argument('--rl', action='store_true', 
                         help='Use Reinforcement Learning AI')
    ai_group.add_argument('--optimal', action='store_true',
                         help='Use Optimal Strategy AI')
    
    # Training arguments
    train_group = parser.add_argument_group('Training Options')
    train_group.add_argument('--train', action='store_true',
                            help='Train RL AI before playing/evaluation')
    train_group.add_argument('--episodes', type=int, default=1000,
                            help='Number of training episodes (default: 1000)')
    
    # Evaluation arguments
    eval_group = parser.add_argument_group('Evaluation Options')
    eval_group.add_argument('--evaluate', action='store_true',
                           help='Run AI evaluation tournament')
    eval_group.add_argument('--num-games', type=int, default=300,
                           help='Number of evaluation games per AI pair (default: 300)')
    eval_group.add_argument('--show-games', action='store_true',
                           help='Show detailed progress of individual games')
    
    return parser

def create_ai_instance(args: argparse.Namespace, game: DiceGame) -> Optional[RLDiceAI]:
    """Create and configure the RL AI instance if needed."""
    if not (args.rl or args.train):
        return None
        
    rl_ai = RLDiceAI(game.moves)
    if args.train:
        print(Fore.CYAN + f"Training RL AI for {args.episodes} episodes..." + Style.RESET_ALL)
        rl_ai.train(args.episodes)
        print(Fore.GREEN + "Training complete!" + Style.RESET_ALL)
    return rl_ai

def setup_ai_for_evaluation(args: argparse.Namespace, game: DiceGame, rl_ai: Optional[RLDiceAI]):
    """Configure the AI instance for evaluation mode."""
    if args.rl and rl_ai:
        print(Fore.CYAN + "Evaluating RL AI..." + Style.RESET_ALL)
        return rl_ai
    elif args.optimal:
        print(Fore.CYAN + "Evaluating Optimal AI..." + Style.RESET_ALL)
        return OptimalAI(game.moves)
    else:
        print(Fore.CYAN + "Evaluating MinMax AI..." + Style.RESET_ALL)
        return MinMaxAI(game.moves)

def main():
    """Main entry point for the D4sky Dice Game."""
    # Parse command-line arguments
    parser = setup_argument_parser()
    args = parser.parse_args()
    
    # Initialize game and AI
    game = DiceGame()
    rl_ai = create_ai_instance(args, game)
    
    if args.evaluate:
        # Evaluation mode: AI tournament
        ai_to_use = setup_ai_for_evaluation(args, game, rl_ai)
        evaluator = GameEvaluator(game.moves, args.show_games, rl_ai=ai_to_use)
        evaluator.evaluate_ais(args.num_games)
    else:
        # Play mode: Human vs AI
        game = DiceGame(use_rl=args.rl, use_optimal=args.optimal)
        if args.rl and rl_ai:
            game.ai = rl_ai
        game.start_game()

if __name__ == "__main__":
    main()
