# AI Chess Engine with Reinforcement Learning and Minimax Algorithm

- Bereket Tilahun UGR/9703/12
- Duressa Jemal UGR/3937/12
- Yeabsira Driba UGR/4951/12
- Beniyam Alemu UGR/4689/12


## Description

This repository contains an AI-powered chess engine that leverages Reinforcement Learning (RL) techniques, specifically Q-learning, combined with the Minimax algorithm for decision making. The project is built using Python and Pygame for visual representation of the chessboard and pieces. The AI engine evaluates board states and makes optimal moves, continuously improving its performance through RL.

## Features

- **Reinforcement Learning (RL)**: Implements Q-learning to train the AI engine, allowing it to learn from its experiences and improve its performance over time.
- **Chess Piece Movement**: Implements movement rules for all chess pieces (Pawn, Rook, Knight, Bishop, Queen, King).
- **Pawn Promotion**: Automatically promotes pawns to a chosen piece when they reach the promotion row.
- **Castling**: Implements castling rules for the king and rook.
- **Check and Checkmate Detection**: Detects and handles check and checkmate conditions.
- **Minimax Algorithm**: Uses the minimax algorithm with alpha-beta pruning for strategic decision making.
- **Threefold Repetition**: Detects and handles the threefold repetition rule for draws.
- **Graphical Interface**: Uses Pygame to display the board and pieces, and to interact with the game.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/bekiTil/AI-Chess_Engine_with_Reinforcement_Learning_and_Minimax_Algorithm.git
   cd AI-Chess_Engine_with_Reinforcement_Learning_and_Minimax_Algorithm
## Usage

Run the main script to start the game:

```bash
python main.py
```
The game will start with the player controlling the white pieces. Click on a piece to select it and then click on a valid square to move it.

The AI will automatically make moves for the black pieces.

The game will display a message when it ends due to checkmate, stalemate, or threefold repetition

## Reinforcement Learning in Chess Engine

### Q-Learning Model

The Q-learning model is a neural network that learns to evaluate board states by updating its knowledge based on the outcomes of moves. The network architecture is as follows:

- **Input Layer:** 64 neurons, one for each square on the board.
- **Hidden Layers:** Two hidden layers with 128 neurons each.
- **Output Layer:** 1 neuron, representing the evaluation score of the board state.

#### Training Process

- **State Representation:** The board state is converted into a tensor representation, where each piece type and position is encoded.
- **Evaluation:** The neural network evaluates the board state and assigns a score.
- **Update Rule:** Q-values are updated using the Q-learning formula:

   **Update Rule: Q-values are updated using the Q-learning formula:**


$$
Q(s,a) \leftarrow Q(s,a) + \alpha \left[ r + \gamma \max_{a'} Q(s',a') - Q(s,a) \right]
$$

where:
- $\alpha$ is the learning rate.
- $r$ is the reward obtained after taking action $a$.
- $\gamma$ is the discount factor.
- $s$ and $s'$ are the current and next states, respectively.
- $a$ and $a'$ are the current and next actions, respectively.



#### Learning Curve

The effectiveness of the RL training process can be visualized using the learning curve, which plots the loss values over training steps. A reduction in loss indicates that the AI is improving its evaluation of board states and making better decisions over time.

- **Initial Learning Phase:** In the initial phase, the loss decreases rapidly, indicating that the model is quickly learning the basics of board evaluation.
- **Extended Learning Phase:** In the extended learning phase, the loss stabilizes, showing that the model is refining its strategies and making more nuanced decisions.

  ![Visualization 1](https://github.com/bekiTil/AI-Chess_Engine_with_Reinforcement_Learning_and_Minimax_Algorithm/blob/main/Visualization1.png)
  ![Visualization 2](https://github.com/bekiTil/AI-Chess_Engine_with_Reinforcement_Learning_and_Minimax_Algorithm/blob/main/Visualization2.png )


### Minimax Algorithm with Alpha-Beta Pruning

The minimax algorithm is used to simulate potential future moves and evaluate the best possible move for the AI. It is combined with Q-learning to refine the AI's decision-making process:

- **Maximizing Player:** The AI aims to maximize its evaluation score.
- **Minimizing Player:** The opponent aims to minimize the AI's evaluation score.
- **Alpha-Beta Pruning:** Optimizes the minimax algorithm by eliminating branches that cannot affect the final decision.

## Project Structure

- **main.py:** Entry point for the game. Sets up the board and handles the game loop.
- **Board.py:** Contains the Board class, which manages the state of the chessboard.
- **Piece.py:** Contains the Piece class and its subclasses (Pawn, Rook, Knight, Bishop, Queen, King), implementing movement rules and other piece-specific logic.
- **ChessEngine.py:** Contains the Q-learning model and functions for board evaluation, Q-learning updates, and the minimax algorithm.


