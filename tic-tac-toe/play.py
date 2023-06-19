from utils.main_functions import main_loop, get_coordinates

if __name__ == "__main__": 

    # To establish the playing area in the game, begin by clicking on all four corners of the screen using your mouse. 
    # Then the game will start.
    board = get_coordinates()
    game = main_loop(board)