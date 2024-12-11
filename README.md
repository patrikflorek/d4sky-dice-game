# D4sky Dice Game

A strategic dice game where players can compete against different AI opponents (MinMax, Reinforcement Learning, or Optimal Strategy) or watch AIs play against each other. The goal is to reach or get as close as possible to 21 points without exceeding it.

## Requirements

- Python 3.12+
- PyTorch 2.1.0+
- NumPy 1.26.0+
- Colorama 0.4.6

## Installation

```bash
# Create and activate virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Command-Line Interface

The game provides a comprehensive CLI with various options for playing, training, and evaluating AI opponents:

### Basic Usage

```bash
# Play against MinMax AI (default)
python main.py

# Play against Optimal AI
python main.py --optimal

# Play against RL AI
python main.py --rl
```

### Training Options

```bash
# Train RL AI with default settings (1000 episodes)
python main.py --train

# Train RL AI with custom episodes
python main.py --train --episodes 2000

# Train and immediately play against the trained AI
python main.py --rl --train --episodes 2000
```

### Evaluation Options

```bash
# Evaluate a specific AI
python main.py --rl --evaluate      # Evaluate RL AI
python main.py --optimal --evaluate # Evaluate Optimal AI

# Custom evaluation settings
python main.py --evaluate --num-games 500    # Run more evaluation games
python main.py --evaluate --show-games       # Show detailed game progress
python main.py --rl --train --episodes 2000 --evaluate  # Train and evaluate RL AI
```

## Game Rules

- Each turn, you can only choose from the available moves shown
- The moves depend on your current dice face
- If you exceed 21 points, you lose
- Getting exactly 21 points is an instant win

### Available Moves
From each face, you can move to:
- Face 1: [2, 3, 4, 5]
- Face 2: [1, 3, 4, 6]
- Face 3: [1, 2, 5, 6]
- Face 4: [1, 2, 5, 6]
- Face 5: [1, 3, 4, 6]
- Face 6: [2, 3, 4, 5]

## AI Implementations

### 1. MinMax AI
- Uses the MinMax algorithm with position caching
- Deterministic strategy optimized for perfect play
- Evaluates all possible future game states
- Computationally intensive but guarantees optimal play

### 2. Optimal Strategy AI
- Implements a hand-crafted strategy based on game theory
- Uses strategic principles and position evaluation
- Features:
  - Recognition of winning positions
  - Safe position targeting
  - Defensive play considerations
  - Face control strategy
- Computationally efficient with near-optimal play

### 3. Reinforcement Learning AI
- Implements Deep Q-Learning (DQN) with curriculum learning
- Features:
  - Experience replay with expert demonstrations
  - Curriculum learning phases
  - Optimal AI guidance during training
  - Dynamic exploration strategy
- Neural network architecture:
  - Input: Current sum and dice face
  - Hidden layers: 2 x 128 neurons
  - Output: Q-values for possible actions
- Training improvements:
  - Initialized with expert knowledge from Optimal AI
  - Progressive reduction in expert guidance
  - Enhanced reward structure
  - Improved exploration strategy

## Optimal Strategy

The game has a deterministic optimal strategy based on maintaining winning positions:

### Winning Positions
These positions allow winning in one move:
- Sum 15-17 with face 6
- Sum 15-18 with faces 3, 4, 5
- Sum 15-19 with face 2
- Sum 16-19 with face 1

### Strategic Principles
1. **Direct Win**: If you can reach 21, take it
2. **Safe Positions**: Aim for these combinations:
   - Sum 16-17 with faces 1 or 2
   - Sum 15 with faces 3, 4, 5, or 6
3. **Defensive Play**:
   - Avoid faces that give opponent winning moves
   - Force opponent into faces with limited high-value moves
4. **Face Control**:
   - Faces 3 and 4 offer better defensive options
   - Maintain favorable face-sum combinations

## Project Structure

```
dice_game/
├── dice_game/
│   ├── __init__.py
│   ├── min_max_ai.py  # MinMax AI with alpha-beta pruning
│   ├── optimal_ai.py  # Optimal strategy AI with strategic constants
│   ├── rl_ai.py       # Reinforcement Learning AI with curriculum learning
│   ├── game.py        # Core game logic and player management
│   ├── models.py      # Game state and data models
│   ├── display.py     # Configurable terminal display system
│   └── evaluator.py   # Comprehensive AI evaluation system
├── main.py            # Command-line interface
├── requirements.txt
└── README.md
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Development

The project follows modern Python best practices:
- Type hints and dataclasses throughout the codebase
- Modular design with clear separation of concerns
- Comprehensive command-line interface
- Configurable display system with customizable themes
- Extensive evaluation capabilities
- Robust error handling and input validation
- Detailed documentation with examples
- Strategic constants and configuration options

### Display System Features
- Customizable dot characters and colors
- Configurable spacing and borders
- Support for different display themes
- Input validation and error handling
- Clear ASCII art representation

### Evaluation System Features
- Head-to-head AI comparisons
- Detailed performance statistics
- Configurable evaluation parameters
- Support for custom game counts
- Progress tracking and reporting

### Game System Features
- Robust player input validation
- Clear error messages and prompts
- Configurable game parameters
- Type-safe game state management
- Enhanced user experience with colored output
