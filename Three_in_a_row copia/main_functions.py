# Main functions used for the three in a row dobot game

import math
import cv2
import numpy as np


def get_coordinates():
    """
    Retrieves the coordinates of four points selected by the user on a video stream.

    Returns:
        A list of four coordinate tuples [(x1, y1), (x2, y2), (x3, y3), (x4, y4)].
    """
    points = []

    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))

    # Open a video capture object
    cap = cv2.VideoCapture(0)

    while True:
        # Capture a frame from the video stream
        ret, frame = cap.read()

        # Display the frame in a window
        cv2.imshow('frame', frame)

        # Set the mouse callback function on the window
        cv2.setMouseCallback('frame', mouse_callback)

        # Exit the loop if four points have been selected
        if len(points) == 4:
            cv2.destroyAllWindows()  # Close the window before breaking out of the loop
            cv2.waitKey(1)  # Process events to allow interruption
            break

        # Exit the loop if the user presses the 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Define the source points as a numpy array
    src_points = np.float32(points)

    # Define the destination points as a numpy array
    dst_points = np.float32([[0, 0], [300, 0], [0, 300], [300, 300]])

    output_listas = [list(tupla) for tupla in points]
    return output_listas



def is_game_over(board):
    """
    Checks if the game is over by examining the board for a winning condition.

    Args:
        board: A 3x3 list representing the game board.

    Returns:
        True if there is a winner, False otherwise.
    """

    # Check rows
    for row in board:
        if len(set(row)) == 1 and row[0] != 0:
            return True

    # Check columns
    for col in range(3):
        if len(set([board[row][col] for row in range(3)])) == 1 and board[0][col] != 0:
            return True

    # Check diagonals
    if len(set([board[i][i] for i in range(3)])) == 1 and board[0][0] != 0:
        return True
    
    if len(set([board[i][2-i] for i in range(3)])) == 1 and board[0][2] != 0:
        return True

    return False


def check_winner(board):
    """
    Checks the board to see if there's a winner.

    Args:
        board: A 3x3 list representing the game board.

    Returns:
        The winning player (1 or 2) if there's a winner, or 0 if no winner.
    """

    # Check rows
    for row in board:
        if row.count(row[0]) == len(row) and row[0] != 0:
            return row[0]

    # Check columns
    for col in range(len(board[0])):
        if all(board[row][col] == board[0][col] and board[0][col] != 0 for row in range(len(board))):
            return board[0][col]

    # Check diagonals
    if all(board[i][i] == board[0][0] and board[0][0] != 0 for i in range(len(board))) or all(board[i][len(board)-1-i] == board[0][len(board)-1] and board[0][len(board)-1] != 0 for i in range(len(board))):
        return board[0][0]

    return 0


def get_available_moves(board, player):
    """
    Returns a list of available moves and the position of the current player's tokens.

    Args:
        board: A 3x3 list representing the game board.
        player: The player's token (1 or 2).

    Returns:
        A tuple containing a list of player positions and a list of available moves.
    """

    moves = []
    player_positions = []

    for i in range(len(board)):
        for j in range(len(board[i])):
            if board[i][j] == player:
                player_positions.append((i, j))
            if board[i][j] == 0:
                moves.append((i, j))

    return player_positions, moves



def minimax(board, depth, maximizing_player, player):
    """
    Minimax algorithm with alpha-beta pruning.

    Args:
        board: A 3x3 list representing the game board.
        depth: The current depth in the minimax algorithm.
        maximizing_player: A boolean indicating if the current player is the maximizing player.
        player: The player's token (1 or -1).

    Returns:
        A tuple containing the best score and the best move for the current player.
    """
    winner = check_winner(board)
    if winner == player:
        return 10 - depth, None
    elif winner == player * -1:
        return depth - 10, None
    elif len(get_available_moves(board, player)[1]) == 0:
        return 0, None

    if maximizing_player:
        best_score = -math.inf
        for i, j in get_available_moves(board, player)[1]:
            new_board = [row[:] for row in board]
            new_board[i][j] = player
            score = minimax(new_board, depth+1, False, player)[0]
            if score > best_score:
                best_score = score
                best_move = (i, j)
        return best_score, best_move
    else:
        best_score = math.inf
        for i, j in get_available_moves(board, player*-1)[1]:
            new_board = [row[:] for row in board]
            new_board[i][j] = player*-1
            score = minimax(new_board, depth+1, True, player)[0]
            if score < best_score:
                best_score = score
                best_move = (i, j)
        return best_score, best_move


def get_best_move(board, player):
    """
    Returns the best move for the given player.

    Args:
        board: A 3x3 list representing the game board.
        player: The player's token (1 or -1).

    Returns:
        A tuple representing the best move for the player.
    """
    player_positions, available_moves = get_available_moves(board, player)
    best_score = -math.inf
    token_pos = None
    best_pos = None
    
    for i, j in available_moves:
        new_board = [row[:] for row in board]
        new_board[i][j] = player
        if is_game_over(new_board):
            return (player_positions[0], (i, j))
        score = minimax(new_board, 0, False, player)[0]
        if score > best_score:
            best_pos = (i, j)
            best_score = score

    # get the current position of a token and the best position to move it
    if len(player_positions) == 3:
        for pos in player_positions:
            if pos != best_pos:
                token_pos = pos
                break
        best_move = (token_pos, best_pos)
    else:
        best_move = (None, best_pos)

    return best_move


def get_square_number(position):
    """
    Returns the cell number (1-9) where the token is placed by the AI.

    Args:
        position: A tuple (row, col) representing the position of the token (0-indexed).

    Returns:
        The cell number (1-9) corresponding to the given position.
    """

    row, col = position
    return row * 3 + col + 1


def update_board1(board, end_square):
    """
    Returns the updated board when a movement by the AI has been done taking the token from outside the board and placing it in a cell.

    Args:
        board: A 3x3 list representing the game board.
        end_square: The cell number (1-9) where the token is placed.

    Returns:
        The updated game board after the movement.
    """

    end_row, end_col = (end_square - 1) // 3, (end_square - 1) % 3
    board[end_row][end_col] = 2
    return board


def update_board2(board, start_square, end_square):
    """
    Returns the updated board when a movement by the AI has been done taking the token from one cell and placing it in another cell.

    Args:
        board: A 3x3 list representing the game board.
        start_square: The cell number (1-9) where the token is currently located.
        end_square: The cell number (1-9) where the token is moved to.

    Returns:
        The updated game board after the movement.
    """

    start_row, start_col = (start_square - 1) // 3, (start_square - 1) % 3
    end_row, end_col = (end_square - 1) // 3, (end_square - 1) % 3
    board[start_row][start_col] = 0
    board[end_row][end_col] = 2
    return board


def check_winning_move(player, result):
    """
    Checks if there is a winning move available for the given player.

    Args:
        player: The player value (1 or 2) for whom to check the winning move.
        result: A 3x3 list representing the game board.

    Returns:
        The position (row, column) of the winning move if available, None otherwise.
    """

    # Check rows
    for row in result:
        if row.count(player) == 2 and row.count(0) == 1:
            col = row.index(0)
            return (result.index(row), col)

    # Check columns
    for i in range(3):
        col = [result[j][i] for j in range(3)]
        if col.count(player) == 2 and col.count(0) == 1:
            row = col.index(0)
            return (row, i)

    # Check diagonals
    diagonal1 = [result[i][i] for i in range(3)]
    diagonal2 = [result[i][2 - i] for i in range(3)]
    if diagonal1.count(player) == 2 and diagonal1.count(0) == 1:
        index = diagonal1.index(0)
        return (index, index)
    if diagonal2.count(player) == 2 and diagonal2.count(0) == 1:
        index = diagonal2.index(0)
        return (index, 2 - index)

    return None


def check_blocking_move(player, result):
    """
    Checks if there is a final winning move from the opponent and blocks it.

    Args:
        player: The player value (1 or 2) for whom to check the blocking move.
        result: A 3x3 list representing the game board.

    Returns:
        The position (row, column) of the blocking move if available, None otherwise.
    """

    return check_winning_move(3 - player, result)
