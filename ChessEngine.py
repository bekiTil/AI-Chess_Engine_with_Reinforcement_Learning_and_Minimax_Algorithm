import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from Board import Board
from Piece import Rook, Knight, Bishop, Queen, King, Pawn  # Ensure correct imports

# Define the Q-learning Model
class QNetwork(nn.Module):
    def __init__(self):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(64, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Initialize the model, loss function, optimizer, and plot data
model = QNetwork()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

losses = []  # List to store loss values

# Convert board to tensor
def board_to_tensor(board):
    tensor = torch.zeros(64)
    piece_values = {
        Pawn: 1,
        Knight: 3,
        Bishop: 3,
        Rook: 5,
        Queen: 9,
        King: 0
    }
    for row in range(8):
        for col in range(8):
            piece = board.get_piece(row, col)
            if piece:
                index = row * 8 + col
                if piece.color == 'white':
                    tensor[index] = piece_values[type(piece)]
                else:
                    tensor[index] = -piece_values[type(piece)]
    return tensor

# Evaluate the board using the Q-learning model
def evaluate_board(board):
    board_tensor = board_to_tensor(board).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        score = model(board_tensor).item()
    return score

# Q-learning update function
def q_learning_update(board, reward, next_board):
    current_state = board_to_tensor(board).unsqueeze(0)
    next_state = board_to_tensor(next_board).unsqueeze(0)

    current_q_value = model(current_state)
    with torch.no_grad():
        next_q_value = model(next_state)
    
    target_q_value = reward + 0.9 * next_q_value  # Discount factor gamma = 0.9
    
    optimizer.zero_grad()
    loss = criterion(current_q_value, target_q_value)
    loss.backward()
    optimizer.step()
    
    losses.append(loss.item())  # Record the loss value

# Minimax algorithm with Q-learning updates
def minimax(board, depth, alpha, beta, maximizing_player, train=False):
    if depth == 0 or board.is_game_over:
        return evaluate_board(board)

    if maximizing_player:
        max_eval = float('-inf')
        for piece, move in board.get_all_player_moves(board.get_current_player_color()):
            piece.move(move, board, True)
            eval = minimax(board, depth - 1, alpha, beta, False, train)
            if train:
                reward = -eval  # Simple reward based on evaluation
                q_learning_update(board, reward, board)  # Update Q-values
            board.undo_move()
            max_eval = max(max_eval, eval)
            alpha = max(alpha, eval)
            if beta <= alpha:
                break
        return max_eval
    else:
        min_eval = float('inf')
        for piece, move in board.get_all_player_moves(board.get_current_player_color()):
            piece.move(move, board, True)
            eval = minimax(board, depth - 1, alpha, beta, True, train)
            if train:
                reward = eval  # Simple reward based on evaluation
                q_learning_update(board, reward, board)  # Update Q-values
            board.undo_move()
            min_eval = min(min_eval, eval)
            beta = min(beta, eval)
            if beta <= alpha:
                break
        return min_eval

# Function to get the best move
def get_best_move(board, depth):
    best_score = float('-inf')
    best_move = None
    original_turn = board.turn  # Store the original turn value
    current_player_color = board.get_current_player_color()  # Store the current player color

    valid_moves = board.get_all_player_moves(current_player_color)
    if not valid_moves:
        return None  # Return None when there are no valid moves

    for piece, move in valid_moves:
        piece.move(move, board, True)
        if not board.is_in_check(current_player_color):
            score = minimax(board, depth - 1, float('-inf'), float('inf'), False, train=True)
            if score > best_score:
                best_score = score
                best_move = (piece, move)
        board.undo_move()
        board.turn = original_turn  # Restore the original turn value

    return best_move

# Plot the learning curve
def plot_learning_curve(losses):
    plt.figure(figsize=(10, 5))
    plt.plot(losses, label='Loss')
    plt.xlabel('Training Step')
    plt.ylabel('Loss')
    plt.title('Learning Curve')
    plt.legend()
    plt.show()


