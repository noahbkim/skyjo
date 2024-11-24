import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque, namedtuple
import random
from skyjo import Player, State, Turn, Flip

# Boilerplate code for Deep Q-Learning on Skyjo
# Notes: 
# - Going to start with 2 agents playing against each other (will add more later)
# - Game state idea (may tinker):
#   - 12 numbers for our hand. Hidden cards will be 5.1
#   - Binary mask for which cards are hidden 
#     (this implementation seems better than one-hot encoding because the numbers are not entirely categorical and saves massive space)
#   - May need to add additional feature to encode "column clearedness"
#   - Opponent's expected score (could also be entire 12-card hand, but seems a bit overkill)
#   - 15-vector for the number of each card already discarded

# Things to consider:
# - Reward function (something like: sum of hand, but discount hidden vs. visible and having clear potential)
# - Structure of neural network
# - Similar to ^ but whether to have 2 similar networks for whether to draw phase and depending on draw (probably)

# Define experience tuple structure
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])

class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_size)
        )
    
    def forward(self, x):
        return self.network(x)

class DQNPlayer(Player):
    def __init__(self, epsilon=0.1):
        super().__init__()
        self.state_size = self._calculate_state_size()
        self.action_size = self._calculate_action_size()
        
        # DQN networks
        self.policy_net = DQN(self.state_size, self.action_size)
        self.target_net = DQN(self.state_size, self.action_size)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        # Training parameters
        self.optimizer = optim.Adam(self.policy_net.parameters())
        self.memory = deque(maxlen=10000)
        self.batch_size = 64
        self.gamma = 0.99
        self.epsilon = epsilon
        self.target_update = 10
        self.steps = 0

    def _calculate_state_size(self):
        # State representation:
        # - Our hand (12 cards)
        # - Top discard card
        # - Number of unflipped cards
        # - Current score
        # - Opponent scores
        return 15  # Simplified for this example

    def _calculate_action_size(self):
        # Actions:
        # - Draw card and place (12 positions)
        # - Draw card and flip (12 positions)
        # - Take discard and place (12 positions)
        return 36

    def _encode_state(self, state: State) -> torch.Tensor:
        # Convert game state to tensor
        encoded = []
        
        # Encode hand
        for finger in self.hand:
            encoded.append(finger.card if finger.card is not None else -99)
        
        # Encode discard pile top card
        encoded.append(state.cards.last_discard)
        
        # Encode number of unflipped cards
        encoded.append(self.hand.unflipped_card_count)
        
        # Encode scores
        encoded.append(self.score)
        
        return torch.FloatTensor(encoded).unsqueeze(0)

    def _decode_action(self, action_idx: int, state: State, turn: Turn):
        action_type = action_idx // 12
        position = action_idx % 12
        
        if action_type == 0:  # Draw and place
            card = turn.draw_card()
            turn.place_drawn_card(position)
        elif action_type == 1:  # Draw and flip
            card = turn.draw_card()
            turn.discard_and_flip(position)
        else:  # Take discard and place
            turn.place_from_discard(position)

    def select_action(self, state_tensor: torch.Tensor) -> int:
        if random.random() < self.epsilon:
            return random.randrange(self.action_size)
        
        with torch.no_grad():
            return self.policy_net(state_tensor).max(1)[1].item()

    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return
        
        experiences = random.sample(self.memory, self.batch_size)
        batch = Experience(*zip(*experiences))
        
        state_batch = torch.cat(batch.state)
        action_batch = torch.tensor(batch.action)
        reward_batch = torch.tensor(batch.reward)
        next_state_batch = torch.cat(batch.next_state)
        done_batch = torch.tensor(batch.done)

        # Compute Q(s_t, a)
        current_q_values = self.policy_net(state_batch).gather(1, action_batch.unsqueeze(1))
        
        # Compute V(s_{t+1}) for all next states
        next_q_values = torch.zeros(self.batch_size)
        with torch.no_grad():
            next_q_values = self.target_net(next_state_batch).max(1)[0]
        
        # Compute expected Q values
        expected_q_values = reward_batch + (self.gamma * next_q_values * (1 - done_batch))
        
        # Compute loss
        loss = nn.MSELoss()(current_q_values, expected_q_values.unsqueeze(1))
        
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def flip(self, state: State, action: Flip) -> None:
        # For initial flips, use a simple heuristic
        # Flip cards at positions 0 and 11 (corners)
        action.flip_card(0)
        action.flip_card(11)

    def turn(self, state: State, turn: Turn) -> None:
        # Convert current state to tensor
        current_state = self._encode_state(state)
        
        # Select action
        action = self.select_action(current_state)
        
        # Execute action
        self._decode_action(action, state, turn)
        
        # Get reward (negative of current hand sum)
        reward = -self.hand.sum_flipped_cards()
        
        # Get next state
        next_state = self._encode_state(state)
        
        # Store transition in memory
        self.memory.append(Experience(
            current_state,
            action,
            reward,
            next_state,
            state.is_round_ending
        ))
        
        # Perform optimization step
        self.optimize_model()
        
        # Update target network
        self.steps += 1
        if self.steps % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

def train_dqn_player(episodes=1000):
    # Create players
    dqn_player = DQNPlayer(epsilon=0.1)
    random_players = [Player() for _ in range(2)]  # Add basic players for training
    all_players = [dqn_player] + random_players
    
    # Training loop
    for episode in range(episodes):
        # Play a game
        state = State(all_players)
        state.play()
        
        # Decay epsilon
        dqn_player.epsilon = max(0.01, dqn_player.epsilon * 0.995)
        
        # Print progress
        if episode % 100 == 0:
            print(f"Episode {episode}, Score: {dqn_player.score}, Epsilon: {dqn_player.epsilon:.3f}")

if __name__ == "__main__":
    train_dqn_player() 