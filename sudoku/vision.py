"""Computer Vision module for Sudoku Solver
Gets the sudoku grid from an image and extracts the numbers from it
Returns a matrix of numbers representing the sudoku grid
"""

import cv2
import operator
import numpy as np
from matplotlib import pyplot as plt
from helpers import (show_image, 
                     find_external_contours, 
                     display_points, 
                     display_rects, 
                     distance_between_points)
from ocr import get_board_numbers


def preprocess_img(img, cleaning=True, invert=True):
    """Preprocesses an image to isolate the gridlines.
    Input:
        img: A 2D numpy array representing the image to be processed.
        cleaning: A boolean indicating whether to perform image cleaning.
    Output:
        A 2D numpy array representing the processed image.
    """
    # Kernel size must be positive and odd and the kernel must be square.
    # The bigger the kernel, the more the image will be blurred
    preprocess = cv2.GaussianBlur(img.copy(), (17, 17), 0)

    # cv2.adaptiveThreshold(src, maxValue, adaptiveMethod, thresholdType, blockSize, constant(c))
    # https://docs.opencv.org/4.x/d7/d4d/tutorial_py_thresholding.html
    # We invert binaries (cv2.THRESH_BINARY_INV) because we want the gridlines to be white, not black
    if invert:
        preprocess = cv2.adaptiveThreshold(
            preprocess, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
        )
    else:
        preprocess = cv2.adaptiveThreshold(
            preprocess, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )

    # Clean the image using dilation and erosion
    # https://docs.opencv.org/3.4/db/df6/tutorial_erosion_dilatation.html
    # Dilation adds pixels to the boundaries of objects in an image.
    # Erosion removes pixels on object boundaries.
    # Opening - erosion followed by dilation. It is useful in removing noise.
    if cleaning:
        kernel = np.ones((3, 3), np.uint8)
        preprocess = cv2.erode(preprocess, kernel, iterations=1)
        #show_image(preprocess)
        if invert:
            preprocess = cv2.dilate(preprocess, kernel, iterations=1)
            #show_image(preprocess)
        if not invert:
            _, blackAndWhite = cv2.threshold(preprocess, 127, 255, cv2.THRESH_BINARY_INV)
            nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(blackAndWhite, None, None, None, 8, cv2.CV_32S)
            sizes = stats[1:, -1] #get CC_STAT_AREA component
            img2 = np.zeros((labels.shape), np.uint8)

            for i in range(0, nlabels - 1):
                if sizes[i] >= 50:   #filter small dotted regions
                    img2[labels == i + 1] = 255
        
            preprocess = cv2.bitwise_not(img2)
            #show_image(preprocess)
    return preprocess


def get_corners(img):
    """Finds the 4 extreme corners of the largest contour (sudoku) in the image.
    Input:
        img: A 2D numpy array representing the image.
    Output:
        A list of 4 tuples, each of which contains the x and y coordinates of a corner.
    """
    # Get list of "outer" contours. So if there is one contour enclosing another, only the outermost is given.
    contours, hierarchy = cv2.findContours(
        img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # Sort by area, descending
    # cv2.ContourArea(): Finds area of outermost polygon (largest feature) in img.
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    polygon = contours[0]  # get largest contour

    # Get points of the selected polygon    
    # Bottom right point has the largest (x + y) value
    bottom_right, _ = max(
        enumerate([pt[0][0] + pt[0][1] for pt in polygon]), 
        key=operator.itemgetter(1) # gets index of point
    )
    # Top left point has the smallest (x + y) value
    top_left, _ = min(
        enumerate([pt[0][0] + pt[0][1] for pt in polygon]),
        key=operator.itemgetter(1)
    )
    # Bottom left point has the largest (x - y) value
    bottom_left, _ = max(
        enumerate([pt[0][0] - pt[0][1] for pt in polygon]),
        key=operator.itemgetter(1)
    )
    # Top right point has the smallest (x - y) value
    top_right, _ = min(
        enumerate([pt[0][0] - pt[0][1] for pt in polygon]),
        key=operator.itemgetter(1)
    )

    print("\nSudoku coords: ",
          str(bottom_right),
          str(bottom_left),
          str(top_right),
          str(top_left)
          )

    return [polygon[top_left][0],
            polygon[top_right][0],
            polygon[bottom_right][0],
            polygon[bottom_left][0]]


def crop_sudoku(img, crop_rect):
    """Crops the sudoku from the image.
    Input:
        img: A 2D numpy array representing the image.
        crop_rect: A list of 4 tuples, each of which contains the x and y coordinates of a corner.
    Output:
        A 2D numpy array representing the cropped image.
    """
    # Define corners of the sudoku in the original image
    top_left = crop_rect[0]
    top_right = crop_rect[1]
    bottom_right = crop_rect[2]
    bottom_left = crop_rect[3]
    source_rect = np.array(
        np.array([top_left, bottom_left, bottom_right, top_right], dtype="float32")
    )  # float for perspective transformation

    # get the longest side in the rectangle
    # this will be the new height and width of the image
    side = max([distance_between_points(bottom_right, top_right),
                distance_between_points(top_left, bottom_left),
                distance_between_points(bottom_right, bottom_left),
                distance_between_points(top_left, top_right)]
                )

    # define the destination points to be a perfect square for the new image
    # 4 corners: (0,0), (side,0), (side,side), (0,side)
    dest_square = np.array(
        [[0, 0], [side, 0], [side, side], [0, side]],
        dtype="float32"
        )
    
    # Perspective Transformation - https://docs.opencv.org/3.1.0/da/d6e/tutorial_py_geometric_transformations.html
    # Skew the image by comparing 4 before and after points -- return matrix
    m = cv2.getPerspectiveTransform(source_rect, dest_square)

    # Apply the matrix to the image
    return cv2.warpPerspective(img, m, (int(side), int(side)))


def get_cells(img):
    """Splits the sudoku into 81 equal cells.
    Input:
        img: A 2D numpy array representing the image.
    Output:
        A list of 81 2D numpy arrays representing the cells. 
        Each cell is a tuple of the top left and bottom right coordinates.
    """
    squares = []
    side = img.shape[:1]  # get the length of the side of the image
    side = side[0] / 9 # Split it in 9 to get length of each cell
    for i in range(9):
        for j in range(9):
            p1 = (i * side, j * side)  # top left corner of box
            p2 = ((i + 1) * side, (j + 1) * side)  # bottom right corner of box
            squares.append((p1, p2))
    return squares


def get_sudoku(img):
    """Gets the sudoku from the image.
    Input:
        img: A 2D numpy array representing the image.
    Output:
        A 2D numpy array representing the sudoku extracted from the image.
    """
    # Clean image
    processed_sudoku = preprocess_img(img)
    #show_image(processed_sudoku)

    # Show how the algorithm finds the corners of the sudoku
    #find_external_contours(processed_sudoku)

    # Find the corners of the sudoku (largest contour)
    corners_of_sudoku = get_corners(processed_sudoku)

    # Display the found corners of the sudoku
    #display_points(processed_sudoku, corners_of_sudoku)

    # Crop the sudoku from the image
    cropped_sudoku = crop_sudoku(img, corners_of_sudoku)

    # Display the cropped sudoku
    # show_image(cropped_sudoku)

    # Get the cells of the sudoku    
    squares_on_sudoku = get_cells(cropped_sudoku)

    #show_image(cropped_sudoku)

    cropped_sudoku = preprocess_img(cropped_sudoku, invert=False)
    #display_rects(cropped_sudoku, squares_on_sudoku)

    # Get the numbers on the sudoku
    board = get_board_numbers(cropped_sudoku, squares_on_sudoku, neural_network=True)
    board = np.transpose(np.array(board).reshape((9, 9)))

    return board
