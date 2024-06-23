# AI Chess Engine with Reinforcement Learning and Minimax Algorithm

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
# Usage

Run the main script to start the game:

```bash
python main.py
```
The game will start with the player controlling the white pieces. Click on a piece to select it and then click on a valid square to move it.

The AI will automatically make moves for the black pieces.

The game will display a message when it ends due to checkmate, stalemate, or threefold repetiti
