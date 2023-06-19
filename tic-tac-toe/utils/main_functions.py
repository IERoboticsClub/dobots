# Main functions used for the three in a row dobot game

import math
import cv2
import numpy as np
from collections import defaultdict
from PIL import Image
import DobotDllType as dType

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


def main_loop(coordinates):
    """
    Main loop of the game
    """
    
    height, width = 450, 350


    # Define the four corners of the tic tac toe board in the original frame
    src_points = np.float32(coordinates)


    # Define the four corners of the destination image
    dst_points = np.float32([[0, 0], [width, 0], [0, height], [width, height]])


    # Initialize a list that will contain the boards
    boards = []


    # Initialize a list that will contain the states of the board 
    results = [[[0, 0, 0], [0, 0, 0], [0, 0, 0]]]
    # final state 
    result = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]


    # Define connection status messages
    CON_STR = {
        dType.DobotConnect.DobotConnect_NoError: "DobotConnect_NoError",
        dType.DobotConnect.DobotConnect_NotFound: "DobotConnect_NotFound",
        dType.DobotConnect.DobotConnect_Occupied: "DobotConnect_Occupied",
    }


    # Load Dll
    api = dType.load()


    # Connect Dobot
    state = dType.ConnectDobot(api, "", 115200)[0]
    print("Connect status:", CON_STR[state])


    # Coordinates of the cells of the board 
    #celda_number = the coordinates of the cell where the token is.
    #celda_numberx = the coordinates of the cell just a litlle bit above the token

    celda_1 = [305.8970031738281, 63.05331802368164, -60, 0]
    celda_1x = [305.8970031738281, 63.05331802368164, -45, 0]

    celda_2 = [306.6937561035156, 4.534292221069336, -60, 0]
    celda_2x = [306.6937561035156, 4.534292221069336, -45, 0]

    celda_3 = [307.52264404296875, -53.817909240722656, -60, 0]
    celda_3x = [307.52264404296875, -53.817909240722656, -45, 0]

    celda_4 = [249.85263061523438, 60.961700439453125, -60, 0]
    celda_4x = [249.85263061523438, 60.961700439453125, -45, 0]

    celda_5 = [249.2190704345703, 3.339076280593872, -60, 0]
    celda_5x = [249.2190704345703, 3.339076280593872, -45, 0]

    celda_6 = [249.5429229736328, -56.50636291503906, -60, 0]
    celda_6x = [249.5429229736328, -56.50636291503906, -45, 0]

    celda_7 = [187.45509338378906, 61.00348663330078, -60, 0]
    celda_7x = [187.45509338378906, 61.00348663330078, -45, 0]

    celda_8 = [188.26858520507812, 0.08686444908380508, -60, 0]
    celda_8x = [188.26858520507812, 0.08686444908380508, -45, 0]

    celda_9 = [190.3065643310547, -56.14841842651367, -60, 0]
    celda_9x = [190.3065643310547, -56.14841842651367, -45, 0]


    # Coordendas de los tokens al iniciar el juego
    token_1 = [181.1938934326172, -138.1505889892578, -60, 0]
    token_2 = [212.1025390625, -137.92701721191406, -60, 0]
    token_3 = [245.34996032714844, -138.14825439453125, -60, 0]


    # Coordenadas de home y middle
    middle = [210.09783935546875, -3.7860727310180664, 116, 0]
    home = [41.88152313232422, -250.90823364257812, 116, 0]


    dType.SetQueuedCmdClear(api)
    dType.SetQueuedCmdStartExec(api)
    dType.SetPTPCmd(api, dType.PTPMode.PTPMOVJXYZMode, home[0], home[1], home[2], home[3], isQueued=1)


    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        M = cv2.getPerspectiveTransform(src_points, dst_points)
        transformed_frame = cv2.warpPerspective(frame, M, (width, height))

        # Code to rotate the image 180 degrees
        pil_image = Image.fromarray(transformed_frame)
        transformed_frame = pil_image.rotate(180)
        transformed_frame = np.array(transformed_frame)

        # Convert the transformed frame to grayscale
        gray = cv2.cvtColor(transformed_frame, cv2.COLOR_BGR2GRAY)

        # Adjust the Canny edge detection thresholds
        low_threshold = 50
        high_threshold = 150
        edges = cv2.Canny(gray, low_threshold, high_threshold, apertureSize=3)

        # Find contours in the image
        contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Initialize an empty list to store the coordinates of the squares
        squares = []

        # Loop through the contours
        for cnt in contours:
            # Approximate the contour with a polygon
            approx = cv2.approxPolyDP(cnt, 0.01 * cv2.arcLength(cnt, True), True)
            # If the polygon has four vertices and is convex, it could be a rectangle
            if len(approx) == 4 and cv2.isContourConvex(approx):
                # Calculate the bounding box of the polygon
                x, y, w, h = cv2.boundingRect(approx)
                # Draw a green rectangle around the rectangle
                cv2.rectangle(transformed_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                # Add the coordinates of the rectangle to the list
                squares.append((x, y, x+w, y+h))

        # Sort the squares from left to right and top to bottom
        squares = sorted(squares, key=lambda c: (c[1] // 124) * 3 + (c[0] // 107))


        # Initialize a 3x3 grid of zeros to represent the tic tac toe board
        board = np.zeros((3, 3), dtype=np.int) 

        # Check if there are exactly 9 squares
        if len(squares) != 9:
            exit()
        
        else:
        # Loop through the squares and fill in the corresponding cell of the grid
            for i, square in enumerate(squares):
                row = i // 3
                col = i % 3
                x1, y1, x2, y2 = square
                # Add the square to the grid
                board[row, col] = 0

                # Crop the image to the bounding box of the square
                square_img = transformed_frame[y1:y2, x1:x2]

                # Convert the cropped image to grayscale
                square_gray = cv2.cvtColor(square_img, cv2.COLOR_BGR2GRAY)

                # Aplicar un filtro gaussiano para reducir el ruido
                blur = cv2.GaussianBlur(gray, (5, 5), 0)

                # Apply the Hough circle transform to detect circles
                circles = cv2.HoughCircles(square_gray, cv2.HOUGH_GRADIENT, dp=1, minDist=20, param1=50, param2=12, minRadius=20, maxRadius=30)


                if circles is not None:

                    circles = circles[0]
                    for (x, y, r) in circles:
                        # Draw a green circle around the detected circle
                        cv2.circle(square_img, (int(x), int(y)), int(r), (0, 255, 0), 2)

                        # Extraer la región del círculo
                        circle_region = square_img[int(y-r):int(y+r), int(x-r):int(x+r)]
                        if not circle_region.any():
                            continue
                        circle_hsv = cv2.cvtColor(circle_region, cv2.COLOR_BGR2HSV)
                        
                        
                        # Definir rangos de color para rojo y verde
                        lower_red = (16, 43, 149)
                        upper_red = (22, 127, 177)
                        lower_green = (98, 13, 32)
                        upper_green = (116, 56, 39)
                        
                        # Segmentar el color del círculo utilizando los rangos de color definidos
                        mask_red = cv2.inRange(circle_hsv, lower_red, upper_red)
                        mask_green = cv2.inRange(circle_hsv, lower_green, upper_green)
                        
                        # Contar los píxeles de cada máscara para determinar el color predominante
                        count_red = cv2.countNonZero(mask_red)
                        count_green = cv2.countNonZero(mask_green)
                        
                        # Imprimir el resultado
                        if count_red > count_green:
                            board[row, col] = 1 
                                                
                        else:
                            board[row, col] = 2 
                

            # Change the type of the board from a string to a list
            board_list = board.tolist()

            # Append the all the new boards to a list
            boards.append(board_list)
            freq = defaultdict(int)

            if len(boards) == 15:
                for board in boards:
                    freq[str(board)] += 1

                # Find board with highest frequency
                max_board = max(freq, key=freq.get)
                result = eval(max_board)  # convert string representation back to list

                results.append(result)

                if len(results) >= 2:
                    if results[-1] != results[-2]:

                        token_count = 0

                        for row in range(3):
                            for col in range(3):
                                if result[row][col] == 2:
                                    token_count += 1
                                    
                        if token_count < 3:
                            # Check for a winning move for the player with tokens value 1
                            winning_move = check_winning_move(1, result)
                            
                            
                            if winning_move is not None:
                                position_1 = get_square_number(winning_move)
                                

                            else:
                                # Check for a blocking move for the player with tokens value 1
                                blocking_move = check_blocking_move(1, result)
                                if blocking_move is not None:
                                    position_1 = get_square_number(blocking_move)

                                else:
                                    position_1 = get_square_number(get_best_move(result, 2)[1])
                            result = update_board1(result, position_1)

                            dType.dSleep(1)
                            dType.SetPTPCmd(api, dType.PTPMode.PTPMOVJXYZMode, middle[0], middle[1], middle[2], middle[3], isQueued=1)
                            dType.dSleep(1)

                            if token_count == 0:
                                dType.SetPTPCmd(api, dType.PTPMode.PTPMOVJXYZMode, token_1[0], token_1[1], token_1[2], token_1[3], isQueued=1)

                            if token_count == 1:
                                dType.SetPTPCmd(api, dType.PTPMode.PTPMOVJXYZMode, token_2[0], token_2[1], token_2[2], token_2[3], isQueued=1)

                            if token_count == 2:
                                dType.SetPTPCmd(api, dType.PTPMode.PTPMOVJXYZMode, token_3[0], token_3[1], token_3[2], token_3[3], isQueued=1)

                            dType.dSleep(1)
                            dType.SetEndEffectorSuctionCup(api, 1, 1)
                            dType.dSleep(1)
                            dType.SetPTPCmd(api, dType.PTPMode.PTPMOVJXYZMode, middle[0], middle[1], middle[2], middle[3], isQueued=1)
                            dType.dSleep(1)

                            if position_1 == 1:
                                dType.SetPTPCmd(api, dType.PTPMode.PTPMOVJXYZMode, celda_1[0], celda_1[1], celda_1[2], celda_1[3], isQueued=1)

                            if position_1 == 2:
                                dType.SetPTPCmd(api, dType.PTPMode.PTPMOVJXYZMode, celda_2[0], celda_2[1], celda_2[2], celda_2[3], isQueued=1)

                            if position_1 == 3:
                                dType.SetPTPCmd(api, dType.PTPMode.PTPMOVJXYZMode, celda_3[0], celda_3[1], celda_3[2], celda_3[3], isQueued=1)

                            if position_1 == 4:
                                dType.SetPTPCmd(api, dType.PTPMode.PTPMOVJXYZMode, celda_4[0], celda_4[1], celda_4[2], celda_4[3], isQueued=1)

                            if position_1 == 5:
                                dType.SetPTPCmd(api, dType.PTPMode.PTPMOVJXYZMode, celda_5[0], celda_5[1], celda_5[2], celda_5[3], isQueued=1)

                            if position_1 == 6:
                                dType.SetPTPCmd(api, dType.PTPMode.PTPMOVJXYZMode, celda_6[0], celda_6[1], celda_6[2], celda_6[3], isQueued=1)

                            if position_1 == 7:
                                dType.SetPTPCmd(api, dType.PTPMode.PTPMOVJXYZMode, celda_7[0], celda_7[1], celda_7[2], celda_7[3], isQueued=1)

                            if position_1 == 8:
                                dType.SetPTPCmd(api, dType.PTPMode.PTPMOVJXYZMode, celda_8[0], celda_8[1], celda_8[2], celda_8[3], isQueued=1)
                            
                            if position_1 == 9:
                                dType.SetPTPCmd(api, dType.PTPMode.PTPMOVJXYZMode, celda_9[0], celda_9[1], celda_9[2], celda_9[3], isQueued=1)
                            
                            dType.dSleep(2)
                            dType.SetEndEffectorSuctionCup(api, 0, 1)
                            dType.dSleep(1)

                            if position_1 == 1:
                                dType.SetPTPCmd(api, dType.PTPMode.PTPMOVJXYZMode, celda_1x[0], celda_1x[1], celda_1x[2], celda_1x[3], isQueued=1)

                            if position_1 == 2:
                                dType.SetPTPCmd(api, dType.PTPMode.PTPMOVJXYZMode, celda_2x[0], celda_2x[1], celda_2x[2], celda_2x[3], isQueued=1)

                            if position_1 == 3:
                                dType.SetPTPCmd(api, dType.PTPMode.PTPMOVJXYZMode, celda_3x[0], celda_3x[1], celda_3x[2], celda_3x[3], isQueued=1)

                            if position_1 == 4:
                                dType.SetPTPCmd(api, dType.PTPMode.PTPMOVJXYZMode, celda_4x[0], celda_4x[1], celda_4x[2], celda_4x[3], isQueued=1)

                            if position_1 == 5:
                                dType.SetPTPCmd(api, dType.PTPMode.PTPMOVJXYZMode, celda_5x[0], celda_5x[1], celda_5x[2], celda_5x[3], isQueued=1)

                            if position_1 == 6:
                                dType.SetPTPCmd(api, dType.PTPMode.PTPMOVJXYZMode, celda_6x[0], celda_6x[1], celda_6x[2], celda_6x[3], isQueued=1)

                            if position_1 == 7:
                                dType.SetPTPCmd(api, dType.PTPMode.PTPMOVJXYZMode, celda_7x[0], celda_7x[1], celda_7x[2], celda_7x[3], isQueued=1)

                            if position_1 == 8:
                                dType.SetPTPCmd(api, dType.PTPMode.PTPMOVJXYZMode, celda_8x[0], celda_8x[1], celda_8x[2], celda_8x[3], isQueued=1)
                            
                            if position_1 == 9:
                                dType.SetPTPCmd(api, dType.PTPMode.PTPMOVJXYZMode, celda_9x[0], celda_9x[1], celda_9x[2], celda_9x[3], isQueued=1)

                            dType.SetPTPCmd(api, dType.PTPMode.PTPMOVJXYZMode, middle[0], middle[1], middle[2], middle[3], isQueued=1)
                            
                        else: 
                            position_1, position_2 = get_square_number(get_best_move(result, 2)[0]), get_square_number(get_best_move(result, 2)[1])
                            result = update_board2(result, position_1, position_2)

                            
                            dType.dSleep(1)

                            if position_1 == 1:
                                dType.SetPTPCmd(api, dType.PTPMode.PTPMOVJXYZMode, celda_1[0], celda_1[1], celda_1[2], celda_1[3], isQueued=1)

                            if position_1 == 2:
                                dType.SetPTPCmd(api, dType.PTPMode.PTPMOVJXYZMode, celda_2[0], celda_2[1], celda_2[2], celda_2[3], isQueued=1)

                            if position_1 == 3:
                                dType.SetPTPCmd(api, dType.PTPMode.PTPMOVJXYZMode, celda_3[0], celda_3[1], celda_3[2], celda_3[3], isQueued=1)

                            if position_1 == 4:
                                dType.SetPTPCmd(api, dType.PTPMode.PTPMOVJXYZMode, celda_4[0], celda_4[1], celda_4[2], celda_4[3], isQueued=1)

                            if position_1 == 5:
                                dType.SetPTPCmd(api, dType.PTPMode.PTPMOVJXYZMode, celda_5[0], celda_5[1], celda_5[2], celda_5[3], isQueued=1)

                            if position_1 == 6:
                                dType.SetPTPCmd(api, dType.PTPMode.PTPMOVJXYZMode, celda_6[0], celda_6[1], celda_6[2], celda_6[3], isQueued=1)

                            if position_1 == 7:
                                dType.SetPTPCmd(api, dType.PTPMode.PTPMOVJXYZMode, celda_7[0], celda_7[1], celda_7[2], celda_7[3], isQueued=1)

                            if position_1 == 8:
                                dType.SetPTPCmd(api, dType.PTPMode.PTPMOVJXYZMode, celda_8[0], celda_8[1], celda_8[2], celda_8[3], isQueued=1)
                            
                            if position_1 == 9:
                                dType.SetPTPCmd(api, dType.PTPMode.PTPMOVJXYZMode, celda_9[0], celda_9[1], celda_9[2], celda_9[3], isQueued=1)
                            
                            dType.dSleep(1)
                            dType.SetEndEffectorSuctionCup(api, 1, 1)
                            dType.dSleep(1)
                            dType.SetPTPCmd(api, dType.PTPMode.PTPMOVJXYZMode, middle[0], middle[1], middle[2], middle[3], isQueued=1)
                            dType.dSleep(1)

                            if position_2 == 1:
                                dType.SetPTPCmd(api, dType.PTPMode.PTPMOVJXYZMode, celda_1[0], celda_1[1], celda_1[2], celda_1[3], isQueued=1)

                            if position_2 == 2:
                                dType.SetPTPCmd(api, dType.PTPMode.PTPMOVJXYZMode, celda_2[0], celda_2[1], celda_2[2], celda_2[3], isQueued=1)

                            if position_2 == 3:
                                dType.SetPTPCmd(api, dType.PTPMode.PTPMOVJXYZMode, celda_3[0], celda_3[1], celda_3[2], celda_3[3], isQueued=1)

                            if position_2 == 4:
                                dType.SetPTPCmd(api, dType.PTPMode.PTPMOVJXYZMode, celda_4[0], celda_4[1], celda_4[2], celda_4[3], isQueued=1)

                            if position_2 == 5:
                                dType.SetPTPCmd(api, dType.PTPMode.PTPMOVJXYZMode, celda_5[0], celda_5[1], celda_5[2], celda_5[3], isQueued=1)

                            if position_2 == 6:
                                dType.SetPTPCmd(api, dType.PTPMode.PTPMOVJXYZMode, celda_6[0], celda_6[1], celda_6[2], celda_6[3], isQueued=1)

                            if position_2 == 7:
                                dType.SetPTPCmd(api, dType.PTPMode.PTPMOVJXYZMode, celda_7[0], celda_7[1], celda_7[2], celda_7[3], isQueued=1)

                            if position_2 == 8:
                                dType.SetPTPCmd(api, dType.PTPMode.PTPMOVJXYZMode, celda_8[0], celda_8[1], celda_8[2], celda_8[3], isQueued=1)
                            
                            if position_2 == 9:
                                dType.SetPTPCmd(api, dType.PTPMode.PTPMOVJXYZMode, celda_9[0], celda_9[1], celda_9[2], celda_9[3], isQueued=1)
                            
                            dType.dSleep(2)
                            dType.SetEndEffectorSuctionCup(api, 0, 1)
                            dType.dSleep(1)

                            if position_2 == 1:
                                dType.SetPTPCmd(api, dType.PTPMode.PTPMOVJXYZMode, celda_1x[0], celda_1x[1], celda_1x[2], celda_1x[3], isQueued=1)

                            if position_2 == 2:
                                dType.SetPTPCmd(api, dType.PTPMode.PTPMOVJXYZMode, celda_2x[0], celda_2x[1], celda_2x[2], celda_2x[3], isQueued=1)

                            if position_2 == 3:
                                dType.SetPTPCmd(api, dType.PTPMode.PTPMOVJXYZMode, celda_3x[0], celda_3x[1], celda_3x[2], celda_3x[3], isQueued=1)

                            if position_2 == 4:
                                dType.SetPTPCmd(api, dType.PTPMode.PTPMOVJXYZMode, celda_4x[0], celda_4x[1], celda_4x[2], celda_4x[3], isQueued=1)

                            if position_2 == 5:
                                dType.SetPTPCmd(api, dType.PTPMode.PTPMOVJXYZMode, celda_5x[0], celda_5x[1], celda_5x[2], celda_5x[3], isQueued=1)

                            if position_2 == 6:
                                dType.SetPTPCmd(api, dType.PTPMode.PTPMOVJXYZMode, celda_6x[0], celda_6x[1], celda_6x[2], celda_6x[3], isQueued=1)

                            if position_2 == 7:
                                dType.SetPTPCmd(api, dType.PTPMode.PTPMOVJXYZMode, celda_7x[0], celda_7x[1], celda_7x[2], celda_7x[3], isQueued=1)

                            if position_2 == 8:
                                dType.SetPTPCmd(api, dType.PTPMode.PTPMOVJXYZMode, celda_8x[0], celda_8x[1], celda_8x[2], celda_8x[3], isQueued=1)
                            
                            if position_2 == 9:
                                dType.SetPTPCmd(api, dType.PTPMode.PTPMOVJXYZMode, celda_9x[0], celda_9x[1], celda_9x[2], celda_9x[3], isQueued=1)

                            dType.SetPTPCmd(api, dType.PTPMode.PTPMOVJXYZMode, middle[0], middle[1], middle[2], middle[3], isQueued=1)

                        
                        print("Robot moves here: ", result)
                        results.append(result)
                        
            
                print(result)

                boards = []

                

        cv2.imshow('frame', transformed_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        #if is_game_over(result) == True:
        #    print("GAME OVER")
        #    break

    dType.DisconnectDobot(api)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
