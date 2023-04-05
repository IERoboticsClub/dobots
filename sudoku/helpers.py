# Helper functions for the Sudoku project.

import cv2
import numpy as np
import matplotlib.pyplot as plt


def show_image(img):
    """Displays an image in a window until any key is pressed.
    Input:
        img: A 2D numpy array representing the image to be displayed.
    """
    cv2.imshow("img", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def plot_many_images(images, titles, rows=1, columns=2):
    """Plots multiple images in a single figure.
    Input:
        images: A list of 2D numpy arrays representing the images to be plotted.
        titles: A list of strings representing the titles of the images.
        rows: An integer representing the number of rows of images to be plotted.
        columns: An integer representing the number of columns of images to be plotted.
    """
    for i, image in enumerate(images):
        plt.subplot(rows, columns, i + 1)
        plt.imshow(image, "gray")
        plt.title(titles[i])
        plt.xticks([]), plt.yticks([])  # Hide tick marks
    plt.show()


def display_points(in_img, points, radius=5, colour=(0, 0, 255)):
    """Displays a list of points on an image.
    Input:
        in_img: A 2D numpy array representing the image on which to display the points.
        points: A list of tuples representing the points to be displayed.
        radius: An integer representing the radius of the points to be displayed.
        colour: A tuple representing the colour of the points to be displayed.
    Output:
        A 2D numpy array representing the image with the points displayed.
    """
    img = in_img.copy()
    if len(colour) == 3:
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[2] == 1:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    for point in points:
        img = cv2.circle(img, tuple(int(x) for x in point), radius, colour, -1)
    show_image(img)
    return img


def display_rects(in_img, rects, colour=255):
    """Displays a list of rectangles on an image.
    Input:
        in_img: A 2D numpy array representing the image on which to display the rectangles.
        rects: A list of tuples representing the rectangles to be displayed.
        colour: A tuple representing the colour of the rectangles to be displayed.
    Output:
        A 2D numpy array representing the image with the rectangles displayed.
    """
    img = in_img.copy()
    for rect in rects:
        img = cv2.rectangle(
            img, tuple(int(x) for x in rect[0]), tuple(int(x) for x in rect[1]), colour
        )
    show_image(img)
    # return img


def find_external_contours(processed_image):
    """Finds the external contours of the gridlines in the processed image.
    Input:
        processed_image: A 2D numpy array representing the processed image.
    Output:
        A list of numpy arrays representing the contours of the gridlines.
    """
    # findContours: boundaries of shapes having same intensity
    # https://docs.opencv.org/4.x/d4/d73/tutorial_py_contours_begin.html

    # https://docs.opencv.org/4.x/d9/d8b/tutorial_py_contours_hierarchy.html
    # RETR_EXTERNAL -> gives "outer" contours. So if you have one contour enclosing another, only the outermost is given.
    # RETR_LIST - > gives all contours.
    # RETR_CCOMP and RETR_TREE -> Take hierarchy into account.

    ext_contours, hierarchy = cv2.findContours(
        processed_image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    contours, hierarchy = cv2.findContours(
        processed_image.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
    )

    processed_image = cv2.cvtColor(processed_image, cv2.COLOR_GRAY2RGB)

    # Draw all contours on image in 2px red lines
    all_contours = cv2.drawContours(
        processed_image.copy(),
        contours,
        -1, # draw all contours
        (255, 0, 0,), # red colour
        2, # thickness
    )
    external_contours = cv2.drawContours(
        processed_image.copy(), ext_contours, -1, (255, 0, 0), 2
    )

    plot_many_images([all_contours, external_contours], ["All contours", "External Only"])


def distance_between_points(p2, p1):
    """Finds the distance between two points.
    Input:
        p1: A tuple containing the x and y coordinates of the first point.
        p2: A tuple containing the x and y coordinates of the second point.
    Output:
        A float representing the distance between the two points."""
    
    # scalar distance between a and b
    # d = sqrt((x2 - x1)^2 + (y2 - y1)^2)
    a = p2[0] - p1[0] # x2 - x1
    b = p2[1] - p1[1] # y2 - y1
    return np.sqrt((a ** 2) + (b ** 2))