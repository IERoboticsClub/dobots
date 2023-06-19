from utils.main_functions import main_loop, get_coordinates

if __name__ == "__main__": 
    board = get_coordinates()
    game = main_loop(board)