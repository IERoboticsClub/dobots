from solver import solve_sudoku
from vision import get_sudoku
import cv2

if __name__ == "__main__":
    img = cv2.imread("assets/sudoku2.jpg", cv2.IMREAD_GRAYSCALE)
    sudoku = get_sudoku(img)
    solution = solve_sudoku(sudoku)