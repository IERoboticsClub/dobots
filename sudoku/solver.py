# Helper functions for Sudoku

def valid_row(sudoku_board, row):
  assigned = []
  for value in sudoku_board[row]:
    if value != 0:
      if value in assigned:
        return False
      else:
        assigned.append(value)
  return True

def valid_column(sudoku_board, column):
  assigned = []
  for row in sudoku_board:
    value = row[column]
    if value != 0:
      if value in assigned:
        return False
      else:
        assigned.append(value)
  return True

# Quadrants are numbered from top left to bottom right
def valid_quadrant(sudoku_board, quadrant):
  starting_row = (quadrant // 3) * 3
  starting_column = (quadrant % 3) * 3
  assigned = []
  for row in sudoku_board[starting_row : starting_row + 3]:
    for value in row[starting_column : starting_column + 3]:
      if value in assigned and value != 0:
        return False
      assigned.append(value)
  return True

# Check if a solution is complete (i.e. there are no 0)
def complete_solution(sudoku_board):
  for row in sudoku_board:
    for value in row:
      if value == 0:
        return False
  return True

# Check if a solution is valid (i.e. respects all constraints)
def valid_solution(sudoku_board):
  for i in range(9):
    if not valid_row(sudoku_board, i):
      return False
    if not valid_column(sudoku_board, i):
      return False
    if not valid_quadrant(sudoku_board, i):
      return False
  return True

# Checks that the solution respects all constraints (including values set in the problem) and is complete
def check_solution(puzzle, solution):
  for row in range(9):
    for col in range(9):
      if puzzle[row][col] != 0 and puzzle[row][col] != solution[row][col]:
        return 'Solution is not a valid, changes a fixed value in the puzzle'
  if not valid_solution(solution):
    return 'Solution does not respect the constraints'
  if not complete_solution(solution):
    return 'Solution is only partial'

def pretty_print(sudoku_board):
  for row in sudoku_board:
    print(row)

# Helper functions to get and set the values from the server
def sudoku_string_to_matrix(sudoku):
  sudoku_board = []
  for i in range(9):
    starting_row = i * 9
    row = []
    for i in range(9):
      row.append(int(sudoku[starting_row + i]))
    sudoku_board.append(row)
  return sudoku_board

def sudoku_matrix_to_string(sudoku_board):
  return ''.join(map(lambda row : ''.join(map(str, row)), sudoku_board))

domains = [i for i in range(1,10)] # Define list of possible domains

def get_coords(a):
  col = a%9
  row = a//9
  return col, row


def is_valid(board, col, row, num):
  board[row][col] = num # Insert possible domain
  quadrant = (row//3)*3 + (col//3)
  return valid_column(board, col) and valid_row(board, row) and valid_quadrant(board, quadrant)


# a: position in sodoku board from 0 to 81 (top left to bottom right)
# n: number of backtracks
def backtrack(board, a=0):
  if complete_solution(board) and valid_solution(board):
    return True
  # get column and row of a
  col, row = get_coords(a)
  if board[row][col] != 0: # If value is 0 move to next item in sodoku
    return backtrack(board, a+1)
  else: # If cell is empty (0)
    for i in domains:
      if is_valid(board, col, row, i): # If valid
        board[row][col] = i
        if backtrack(board, a+1):
          return board, a
        board[row][col] = 0
      board[row][col] = 0
    
    global backtracking_count
    backtracking_count += 1
    return False


def solve_sudoku(sudoku_board):

    print('Input puzzle')
    pretty_print(sudoku_board)
    
    global backtracking_count
    backtracking_count = 0
    solution = backtrack(sudoku_board)[0]

    print('Solution')
    pretty_print(solution)
    check_solution(sudoku_board, solution)
    print("Backtracking Count", backtracking_count)

    return solution