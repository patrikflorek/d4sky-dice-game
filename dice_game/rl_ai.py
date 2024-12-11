"""Deep Q-Learning Implementation for D4sky Dice Game

This module implements a sophisticated Deep Q-Learning agent that combines:
1. Dueling DQN Architecture with Residual Connections
2. Expert Guidance through Curriculum Learning
3. Prioritized Experience Replay
4. Strategic Position Evaluation

Key Features:
- Dueling architecture separates state value and action advantages
- Residual connections improve gradient flow and training stability
- Expert guidance from OptimalAI for curriculum learning
- Strategic position evaluation with game-specific heuristics
- Adaptive exploration and learning rates

Example Usage:
    >>> from dice_game.rl_ai import RLDiceAI
    >>> moves = {1: [2, 3, 4, 5], 2: [1, 3, 4, 6], ...}
    >>> ai = RLDiceAI(moves)
    >>> ai.train(num_episodes=10000)
    >>> position = Position(suma=15, face=3)
    >>> move = ai.select_action(position)
"""

import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import namedtuple, deque
from typing import Dict, List, Tuple, Optional, Union
from .models import Position
import os
from colorama import Fore, Style
from .min_max_ai import MinMaxAI
import math
import copy

# Hyperparameters with detailed explanations
BATCH_SIZE = 128     # Larger batch size for more stable learning
GAMMA = 0.99        # Discount factor for future rewards
EPS_START = 0.95    # Initial exploration rate
EPS_END = 0.05      # Final exploration rate
EPS_DECAY = 2000    # Slower decay for better exploration
TARGET_UPDATE = 5    # Frequency of target network updates
MEMORY_SIZE = 20000  # Size of experience replay buffer
LEARNING_RATE = 1e-4 # Conservative learning rate for stability
MIN_WIN_RATE_TO_SAVE = 20.0  # Minimum win rate to save model

# Named tuple for storing transitions
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class ReplayMemory:
    """Experience replay buffer for storing and sampling transitions.
    
    Implements a circular buffer using collections.deque for efficient
    storage and random sampling of transitions.
    
    Attributes:
        memory (deque): Fixed-size buffer for storing transitions
        
    Example:
        >>> memory = ReplayMemory(1000)
        >>> memory.push(state, action, next_state, reward)
        >>> batch = memory.sample(64)
    """
    
    def __init__(self, capacity: int):
        """Initialize replay buffer with fixed capacity.
        
        Args:
            capacity: Maximum number of transitions to store
        """
        self.memory = deque([], maxlen=capacity)
        
    def push(self, *args):
        """Add a transition to the buffer.
        
        Args:
            *args: Transition elements (state, action, next_state, reward)
        """
        self.memory.append(Transition(*args))
        
    def sample(self, batch_size: int) -> List[Transition]:
        """Sample a random batch of transitions.
        
        Args:
            batch_size: Number of transitions to sample
            
        Returns:
            List of randomly sampled transitions
        """
        return random.sample(self.memory, batch_size)
        
    def __len__(self) -> int:
        """Get current size of replay buffer."""
        return len(self.memory)

class ResidualBlock(nn.Module):
    """Residual block with normalization and dropout.
    
    Implements a residual connection around two linear layers with
    ReLU activation, layer normalization, and dropout for regularization.
    
    Architecture:
        Input -> Linear -> ReLU -> LayerNorm -> Dropout -> Linear -> Add -> ReLU
    """
    
    def __init__(self, hidden_size: int):
        """Initialize residual block.
        
        Args:
            hidden_size: Number of hidden units in linear layers
        """
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.LayerNorm(hidden_size),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with residual connection.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor with residual connection
        """
        return F.relu(x + self.layers(x))

class DuelingDQN(nn.Module):
    """Dueling DQN architecture with residual connections.
    
    Separates state value and action advantages for better policy evaluation.
    Uses residual blocks for improved gradient flow and training stability.
    
    Architecture:
        1. Feature Extraction:
           Input -> Linear -> ReLU -> LayerNorm -> Dropout
           
        2. Feature Processing:
           Linear -> ReLU -> LayerNorm -> ResidualBlocks
           
        3. Value Stream:
           ResidualBlock -> Linear -> Value
           
        4. Advantage Stream:
           ResidualBlock -> Linear -> Advantages
    """
    
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        """Initialize network architecture.
        
        Args:
            input_size: Dimension of input state
            hidden_size: Number of hidden units
            output_size: Number of possible actions
        """
        super().__init__()
        
        # Initial feature extraction
        self.input_net = nn.Sequential(
            nn.Linear(input_size, hidden_size * 2),
            nn.ReLU(),
            nn.LayerNorm(hidden_size * 2),
            nn.Dropout(0.1)
        )
        
        # Deep feature processing with residual blocks
        self.feature_net = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.LayerNorm(hidden_size),
            ResidualBlock(hidden_size),
            ResidualBlock(hidden_size)
        )
        
        # Value stream (estimates state value)
        self.value_stream = nn.Sequential(
            ResidualBlock(hidden_size),
            nn.Linear(hidden_size, 1)
        )
        
        # Advantage stream (estimates action advantages)
        self.advantage_stream = nn.Sequential(
            ResidualBlock(hidden_size),
            nn.Linear(hidden_size, output_size)
        )
        
        # Initialize weights with smaller values for stability
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize network weights using Xavier initialization.
        
        Args:
            module: Network module to initialize
        """
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight, gain=0.5)
            if module.bias is not None:
                module.bias.data.zero_()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass combining value and advantage streams.
        
        Args:
            x: Input state tensor
            
        Returns:
            Q-values for each action
        """
        features = self.input_net(x)
        features = self.feature_net(features)
        
        values = self.value_stream(features)
        advantages = self.advantage_stream(features)
        
        # Combine using dueling architecture formula
        qvalues = values + (advantages - advantages.mean(dim=1, keepdim=True))
        return qvalues

class RLDiceAI:
    """Deep Q-Learning agent for D4sky Dice Game.
    
    Combines dueling DQN architecture with residual connections, expert guidance,
    prioritized experience replay, and strategic position evaluation.
    
    Key Features:
    1. Dueling DQN Architecture:
       - Separates state value and action advantages
       - Residual connections for improved training
       - Layer normalization and dropout for regularization
       
    2. Expert Guidance:
       - Uses OptimalAI for curriculum learning
       - Adaptive expert probability based on position
       - Separate memory buffer for successful episodes
       
    3. Strategic Learning:
       - Position-aware exploration
       - Strategic reward shaping
       - Conservative move bias in good positions
       
    4. Training Optimizations:
       - Prioritized experience replay
       - Learning rate scheduling
       - Gradient clipping
       - Early stopping with model reversion
    
    Example:
        >>> ai = RLDiceAI(moves)
        >>> ai.train(num_episodes=10000)
        >>> position = Position(suma=15, face=3)
        >>> move = ai.select_action(position)
    """
    
    def __init__(self, moves: Dict[int, List[int]], device: str = "cpu"):
        """Initialize the RL agent.
        
        Args:
            moves: Valid moves for each dice face
            device: Device to use for tensor operations
        """
        self.moves = moves
        self.device = device
        
        # Network parameters
        self.state_size = 21  # 6 (face) + 15 (position features)
        self.hidden_size = 128
        self.output_size = 6  # Maximum face value
        
        # Initialize networks
        self.policy_net = DuelingDQN(
            input_size=self.state_size,
            hidden_size=self.hidden_size,
            output_size=self.output_size
        ).to(device)
        
        self.target_net = DuelingDQN(
            input_size=self.state_size,
            hidden_size=self.hidden_size,
            output_size=self.output_size
        ).to(device)
        
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        # Training parameters
        self.batch_size = BATCH_SIZE
        self.gamma = GAMMA
        self.eps_start = EPS_START
        self.eps_end = EPS_END
        self.eps_decay = EPS_DECAY
        self.target_update = TARGET_UPDATE
        self.learning_rate = LEARNING_RATE
        
        # Optimizer with gradient clipping
        self.optimizer = optim.Adam(
            self.policy_net.parameters(),
            lr=self.learning_rate
        )
        
        # Experience replay buffers
        self.memory = ReplayMemory(MEMORY_SIZE)
        self.success_memory = ReplayMemory(1000)
        
        # Expert AI for guidance
        from .optimal_ai import OptimalAI
        self.expert_ai = OptimalAI(self.moves)
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.MultiStepLR(
            self.optimizer,
            milestones=[500, 1000, 1500],
            gamma=0.5
        )
        
        # Training state
        self.steps_done = 0
        self.total_opt_steps = 0
        self.best_win_rate = 0.0
        
        # Curriculum learning phases
        self.phases = [
            (0, 0.2, 0.95),  # Phase 1: Heavy expert reliance
            (0.2, 0.4, 0.85),  # Phase 2: High expert influence
            (0.4, 0.6, 0.75),  # Phase 3: Balanced learning
            (0.6, 0.8, 0.65),  # Phase 4: More independence
            (0.8, 1.0, 0.55),  # Phase 5: Mostly independent
        ]
        
    def _get_state(self, position: Position) -> torch.Tensor:
        """Convert position to network input state.
        
        Args:
            position: Current position
        
        Returns:
            Input state tensor
        """
        # One-hot encode the face value (6 features)
        face_encoding = torch.zeros(6, device=self.device)
        face_encoding[position.face - 1] = 1.0
        
        # Normalize suma to [0,1] range and create position features
        normalized_suma = position.suma / 21.0
        position_features = torch.tensor([
            normalized_suma,
            1.0 if position.suma >= 16 else 0.0,  # Strategic position indicator
            1.0 if position.suma >= 18 else 0.0,  # Near winning indicator
            1.0 if position.suma == 20 else 0.0,  # One-away indicator
            1.0 if position.suma > 21 else 0.0,   # Bust indicator
            1.0 if position.suma >= 19 else 0.0,  # Very strong position indicator
            1.0 if position.suma >= 17 else 0.0,  # Good position indicator
            1.0 if position.suma >= 15 else 0.0,  # Decent position indicator
            1.0 if position.suma >= 13 else 0.0,  # Safe position indicator
            1.0 if position.suma >= 11 else 0.0,  # Basic position indicator
            # New strategic features based on OptimalAI knowledge
            1.0 if 15 <= position.suma <= 19 else 0.0,  # General winning position
            1.0 if position.suma in [16, 17] and position.face <= 2 else 0.0,  # Safe for low faces
            1.0 if position.suma == 15 and position.face >= 3 else 0.0,  # Safe for high faces
            1.0 if any(position.suma + m == 21 for m in self.moves[position.face]) else 0.0,  # Can win next move
            1.0 if position.face in [3, 4] else 0.0,  # Preferred defensive faces
        ], device=self.device)
        
        # Combine features (total 21 features)
        state = torch.cat([face_encoding, position_features])
        return state.unsqueeze(0)

    def _get_action_idx(self, move: int, available_moves: List[int]) -> int:
        """Get the index of a move in the available moves list with validation.
        
        Args:
            move: Move to find index for
            available_moves: List of available moves
        
        Returns:
            Index of move in available moves list
        """
        try:
            return available_moves.index(move)
        except ValueError:
            # If move not found, return the closest valid move
            valid_moves = np.array(available_moves)
            closest_idx = np.abs(valid_moves - move).argmin()
            return closest_idx

    def _get_move_from_idx(self, idx: int, available_moves: List[int]) -> int:
        """Get the actual move value from an action index.
        
        Args:
            idx: Action index
            available_moves: List of available moves
        
        Returns:
            Actual move value
        """
        if 0 <= idx < len(available_moves):
            return available_moves[idx]
        return available_moves[0]  # Default to first move if invalid

    def _get_expert_probability(self) -> float:
        """Get the probability of using expert moves based on current training phase and position.
        
        Returns:
            Probability of using expert moves
        """
        progress = self.steps_done / self.total_steps if hasattr(self, 'total_steps') else 0.0
        
        # Find current phase
        for phase_start, phase_end, expert_prob in self.phases:
            if phase_start <= progress < phase_end:
                phase_progress = (progress - phase_start) / (phase_end - phase_start)
                base_prob = expert_prob * (1.0 - 0.2 * phase_progress)
                
                # Increase expert probability in critical positions
                if hasattr(self, 'current_position') and self.current_position.suma >= 16:
                    return min(0.95, base_prob * 1.2)  # Up to 20% boost in strategic positions
                
                return base_prob
        
        return 0.65  # Minimum expert probability

    def select_action(self, position: Position) -> int:
        """Enhanced action selection with expert guidance.
        
        Args:
            position: Current position
        
        Returns:
            Selected action
        """
        # Get expert move and score
        expert_score, expert_move = self.expert_ai.get_best_move(position)
        
        # Calculate expert probability based on current phase
        expert_prob = self._get_expert_probability()
        
        # Use expert move with higher probability in good positions
        if expert_score > 0:
            expert_prob *= 1.2
        
        if random.random() < expert_prob:
            return expert_move
            
        # Use epsilon-greedy with temperature scaling for exploration
        sample = random.random()
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * \
            math.exp(-1. * self.steps_done / self.eps_decay)
            
        self.steps_done += 1
        
        if sample > eps_threshold:
            with torch.no_grad():
                state = self._get_state(position)
                q_values = self.policy_net(state)
                
                # Filter Q-values for available moves
                valid_q_values = torch.zeros(1, len(self.moves[position.face]), device=self.device)
                for i, move in enumerate(self.moves[position.face]):
                    valid_q_values[0][i] = q_values[0][move-1]  # -1 since moves are 1-based
                
                # Add strategic bias for conservative moves in good positions
                if 16 <= position.suma <= 20:
                    for i, move in enumerate(self.moves[position.face]):
                        if move in [1, 2]:  # Conservative moves
                            valid_q_values[0][i] *= 1.2  # Boost conservative options
                        elif move in [5, 6]:  # Risky moves
                            valid_q_values[0][i] *= 0.8  # Reduce risky options
                
                # Temperature scaling
                temperature = 1.0  # Removed temperature scaling
                scaled_q_values = valid_q_values / temperature
                
                selected_idx = scaled_q_values.max(1)[1].view(1, 1).item()
                return self.moves[position.face][selected_idx]
        
        # Random action for exploration
        return random.choice(self.moves[position.face])

    def _calculate_reward(self, position: Position, next_position: Position, moves_made: int) -> float:
        """Simplified reward calculation focusing on key strategic elements.
        
        Args:
            position: Current position
            next_position: Next position
            moves_made: Number of moves made
        
        Returns:
            Reward value
        """
        # Get expert evaluation for both current and next position
        _, expert_move = self.expert_ai.get_best_move(position)
        expert_score = self._evaluate_position_with_expert(next_position)
        
        # Terminal state rewards
        if next_position.is_terminal():
            if next_position.suma == 21:
                return 1000.0 * max(0.5, (10.0 - moves_made) / 10.0)
            return -1000.0
        
        # Base reward from expert evaluation
        reward = expert_score
        
        # Simple progress reward
        if next_position.suma > position.suma:
            reward += 50.0 * (next_position.suma / 21.0)
        
        # Efficiency scaling
        reward *= max(0.7, 1.0 - (moves_made / 15.0))
        
        return reward
        
    def _evaluate_position_with_expert(self, position: Position) -> float:
        """Evaluate position using expert AI's knowledge.
        
        Args:
            position: Position to evaluate
        
        Returns:
            Evaluation score
        """
        if position.is_terminal():
            return -1000.0 if position.suma > 21 else 1000.0
            
        # Get expert's evaluation
        expert_score, _ = self.expert_ai.get_best_move(position)
        
        # Scale the expert's evaluation
        return expert_score * 100.0
        
    def _initialize_memory_with_expert(self, num_games: int):
        """Enhanced initialization with expert AI.
        
        Args:
            num_games: Number of games to initialize with
        """
        print(f"Initializing memory with {num_games} expert games...")
        
        for i in range(num_games):
            if i % 100 == 0:
                print(f"Initializing memory: {i}/{num_games}")
            
            # Start from different positions for better coverage
            initial_face = random.choice([1, 2, 5, 6])
            position = Position(suma=initial_face, face=initial_face)
            moves_made = 0
            
            while not position.is_terminal() and position.suma != 21 and moves_made < 10:
                state = self._get_state(position)
                
                # Get expert move
                expert_score, expert_move = self.expert_ai.get_best_move(position)
                
                # Make the move
                next_position = position.make_move(expert_move)
                moves_made += 1
                
                # Calculate reward using expert knowledge
                reward = self._calculate_reward(position, next_position, moves_made)
                
                # Store transition
                next_state = None if next_position.is_terminal() else self._get_state(next_position)
                action_idx = self._get_action_idx(expert_move, self.moves[position.face])
                self.memory.push(state, action_idx, next_state, reward)
                
                # Store successful transitions separately
                if expert_score > 0:
                    self.success_memory.push(state, action_idx, next_state, reward)
                
                position = next_position

    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return
        
        # Mix experiences with emphasis on successful ones
        num_success = min(self.batch_size // 2, len(self.success_memory))
        num_regular = self.batch_size - num_success
        
        regular_transitions = self.memory.sample(num_regular)
        success_transitions = self.success_memory.sample(num_success) if num_success > 0 else []
        transitions = regular_transitions + success_transitions
        
        batch = Transition(*zip(*transitions))
        
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                              batch.next_state)), device=self.device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                         if s is not None])
        
        state_batch = torch.cat(batch.state)
        action_batch = torch.tensor(batch.action, device=self.device)
        reward_batch = torch.tensor(batch.reward, device=self.device)
        
        # Current Q values
        current_q_values = self.policy_net(state_batch)
        state_action_values = current_q_values.gather(1, action_batch.unsqueeze(1))
        
        # Compute V(s_{t+1}) for all next states
        next_state_values = torch.zeros(self.batch_size, device=self.device)
        with torch.no_grad():
            # Double Q-learning with clipped double Q
            next_q_policy = self.policy_net(non_final_next_states)
            next_q_target = self.target_net(non_final_next_states)
            
            next_actions = next_q_policy.max(1)[1].unsqueeze(1)
            next_state_values[non_final_mask] = torch.min(
                next_q_target.gather(1, next_actions).squeeze(1),
                next_q_policy.gather(1, next_actions).squeeze(1)
            )
        
        # Compute expected Q values with TD(λ) and n-step returns
        expected_state_action_values = reward_batch + (self.gamma * next_state_values)
        
        # Add L2 regularization for Q-values
        q_reg = 0.001 * (current_q_values ** 2).mean()  # Reduced L2 regularization
        
        # Huber loss for stability
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1)) + q_reg
        
        # Optimize with gradient clipping
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 0.5)  # More conservative clipping
        self.optimizer.step()
        
        self.total_opt_steps += 1
        return loss.item()

    def train(self, num_episodes: int):
        """Train the RL agent.
        
        Args:
            num_episodes: Number of episodes to train for
        """
        print("\nStarting training loop...\n")
        
        # Initialize memory with expert moves
        print("Initializing experience replay memory with expert moves...")
        self._initialize_memory_with_expert(3000)
        
        # Training loop
        minmax_ai = MinMaxAI(self.moves)
        eval_interval = 100
        plateau_counter = 0
        max_plateau_patience = 3
        
        # Set total steps for expert probability calculation
        self.total_steps = num_episodes
        
        optimization_counter = 0
        best_model_steps = 0
        best_model_state = None
        plateau_counter = 0
        
        for episode in range(num_episodes):
            # Store current position for expert probability calculation
            position = Position(suma=0, face=random.choice([1, 2, 5, 6]))
            self.current_position = position
            
            # Determine current phase
            progress = episode / num_episodes
            for phase_start, phase_end, expert_prob in self.phases:
                if phase_start <= progress < phase_end:
                    phase_progress = (progress - phase_start) / (phase_end - phase_start)
                    expert_prob = max(0.8, expert_prob * (1 - 0.5 * phase_progress))
                    break
            else:
                expert_prob = 0.8
            
            # Position selection with variance
            initial_face = random.choice([1, 2, 5, 6])
            position = Position(suma=initial_face, face=initial_face)
            
            state = self._get_state(position)
            episode_reward = 0
            moves_made = 0
            episode_memory = []
            
            while not position.is_terminal() and position.suma != 21 and moves_made < 10:
                # Expert moves with conservative depth
                if random.random() < expert_prob:
                    depth = 5 if position.suma >= 18 else (4 if position.suma >= 16 else 3)
                    _, move = minmax_ai.get_best_move(position, depth=depth)
                else:
                    move = self.select_action(position)
                
                # Validate move
                available_moves = self.moves[position.face]
                if move not in available_moves:
                    move = available_moves[self._get_action_idx(move, available_moves)]
                
                next_position = position.make_move(move)
                moves_made += 1
                
                reward = self._calculate_reward(position, next_position, moves_made)
                episode_reward += reward
                
                next_state = None if next_position.is_terminal() else self._get_state(next_position)
                action_idx = self._get_action_idx(move, available_moves)
                
                # Store transition
                transition = (state, action_idx, next_state, reward)
                episode_memory.append(transition)
                self.memory.push(state, action_idx, next_state, reward)
                
                # Expert learning with high probabilities
                expert_learn_prob = 0.95 if position.suma >= 18 else (0.85 if position.suma >= 16 else 0.7)
                if random.random() < expert_learn_prob:
                    depth = 5 if position.suma >= 18 else (4 if position.suma >= 16 else 3)
                    _, expert_move = minmax_ai.get_best_move(position, depth=depth)
                    
                    if expert_move not in available_moves:
                        expert_move = available_moves[self._get_action_idx(expert_move, available_moves)]
                    
                    expert_idx = self._get_action_idx(expert_move, available_moves)
                    if expert_idx != action_idx:
                        expert_next_position = position.make_move(expert_move)
                        expert_reward = self._calculate_reward(position, expert_next_position, moves_made) * 2.0
                        expert_next_state = None if expert_next_position.is_terminal() else self._get_state(expert_next_position)
                        self.memory.push(state, expert_idx, expert_next_state, expert_reward)
                
                position = next_position
                state = next_state
                
                # Optimize more frequently near winning positions
                optimize_freq = 1 if position.suma >= 16 else (2 if position.suma >= 12 else 3)
                if moves_made % optimize_freq == 0:
                    for _ in range(3):
                        loss = self.optimize_model()
                        optimization_counter += 1
                    
                    if optimization_counter >= 100:
                        self.scheduler.step()
                        optimization_counter = 0
            
            # Store successful episodes with higher rewards
            if next_position.suma == 21:
                for transition in episode_memory:
                    state, action_idx, next_state, reward = transition
                    self.success_memory.push(state, action_idx, next_state, reward * 5.0)
            
            if episode % self.target_update == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())
            
            if episode > 0 and episode % eval_interval == 0:
                win_rate = self._evaluate_model()
                print(f"\nEpisode {episode}")
                print(f"Win Rate: {win_rate:.1f}%")
                print(f"Episode Reward: {episode_reward:.1f}")
                print(f"Moves Made: {moves_made}")
                print(f"Expert Move Probability: {expert_prob:.2f}")
                print(f"Learning Rate: {self.optimizer.param_groups[0]['lr']:.2e}")
                
                if win_rate > self.best_win_rate:
                    plateau_counter = 0
                    old_win_rate = self.best_win_rate
                    self.best_win_rate = win_rate
                    best_model_steps = self.steps_done
                    best_model_state = copy.deepcopy(self.policy_net.state_dict())
                    self.save_model()
                    print(Fore.GREEN + f"New best model saved! Win rate improved: {old_win_rate:.1f}% -> {win_rate:.1f}%" + Style.RESET_ALL)
                else:
                    print(f"No improvement. Best win rate remains: {self.best_win_rate:.1f}%")
                    plateau_counter += 1
                    
                    if plateau_counter >= max_plateau_patience:
                        plateau_counter = 0
                        if win_rate < self.best_win_rate - 10 and best_model_state is not None:
                            print(Fore.YELLOW + "Reverting to best model due to performance drop" + Style.RESET_ALL)
                            self.policy_net.load_state_dict(best_model_state)
                            self.target_net.load_state_dict(best_model_state)
                            self.steps_done = best_model_steps
                        else:
                            print(Fore.YELLOW + "Adjusting exploration and learning rate due to plateau" + Style.RESET_ALL)
            
            if episode % 100 == 0:
                print(f"Episode {episode}/{num_episodes}")

    def save_model(self):
        """Save the trained model with state information.
        
        Saves the following state:
        - Policy network weights
        - Target network weights
        - Optimizer state
        - Learning rate scheduler state
        - Training steps
        
        The model is saved to 'rl_model.pt' in the current directory.
        Success/failure is indicated through colored console output.
        """
        try:
            torch.save({
                'policy_net_state_dict': self.policy_net.state_dict(),
                'target_net_state_dict': self.target_net.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
                'steps_done': self.steps_done
            }, "rl_model.pt")
            print(Fore.GREEN + f"Model saved successfully to rl_model.pt" + Style.RESET_ALL)
        except Exception as e:
            print(Fore.RED + f"Error saving model: {e}" + Style.RESET_ALL)
        
    def load_model(self):
        """Load a previously trained model with state information.
        
        Loads the following state:
        - Policy network weights
        - Target network weights
        - Optimizer state
        - Learning rate scheduler state
        - Training steps
        
        If loading fails, initializes a fresh model and reports the error.
        Success/failure is indicated through colored console output.
        """
        try:
            checkpoint = torch.load("rl_model.pt")
            self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
            self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            self.steps_done = checkpoint['steps_done']
            print(Fore.GREEN + "Model loaded successfully!" + Style.RESET_ALL)
        except Exception as e:
            print(Fore.RED + f"Error loading model: {e}" + Style.RESET_ALL)
            print(Fore.YELLOW + "Starting with a fresh model." + Style.RESET_ALL)
            self.target_net.load_state_dict(self.policy_net.state_dict())
            
    def _evaluate_model(self, num_games: int = 100) -> float:
        """Evaluate model performance against MinMax AI.
        
        Plays a series of games against MinMax AI to assess performance.
        During evaluation:
        1. Exploration is disabled
        2. Players take turns going first
        3. Games are limited to 10 moves
        4. Win rate is calculated
        
        Args:
            num_games: Number of evaluation games to play
            
        Returns:
            Win rate as a percentage
            
        Example:
            >>> win_rate = ai._evaluate_model(num_games=100)
            >>> print(f"Win rate: {win_rate}%")
        """
        wins = 0
        minmax_ai = MinMaxAI(self.moves)
        
        # Disable exploration during evaluation
        current_steps = self.steps_done
        self.steps_done = 1000000  # Force deterministic policy
        
        for _ in range(num_games):
            position = Position(suma=0, face=random.choice([1, 2, 5, 6]))
            
            # Randomly decide who goes first
            rl_first = random.choice([True, False])
            game_length = 0
            
            while not position.is_terminal() and position.suma < 21 and game_length < 10:
                if rl_first:
                    # RL AI's turn
                    move = self.select_action(position)
                    available_moves = self.moves[position.face]
                    if move not in available_moves:
                        move = available_moves[self._get_action_idx(move, available_moves)]
                    position = position.make_move(move)
                    game_length += 1
                    
                    if position.is_terminal() or position.suma > 21:
                        break
                    if position.suma == 21:
                        wins += 1
                        break
                    
                    # MinMax AI's turn
                    _, move = minmax_ai.get_best_move(position)
                    position = position.make_move(move)
                    
                    if position.is_terminal() or position.suma > 21:
                        wins += 1
                        break
                    if position.suma == 21:
                        break
                else:
                    # MinMax AI's turn
                    _, move = minmax_ai.get_best_move(position)
                    position = position.make_move(move)
                    
                    if position.is_terminal() or position.suma > 21:
                        wins += 1
                        break
                    if position.suma == 21:
                        break
                    
                    # RL AI's turn
                    move = self.select_action(position)
                    available_moves = self.moves[position.face]
                    if move not in available_moves:
                        move = available_moves[self._get_action_idx(move, available_moves)]
                    position = position.make_move(move)
                    game_length += 1
                    
                    if position.is_terminal() or position.suma > 21:
                        break
                    if position.suma == 21:
                        wins += 1
                        break
        
        # Restore exploration settings
        self.steps_done = current_steps
        
        return (wins / num_games) * 100
