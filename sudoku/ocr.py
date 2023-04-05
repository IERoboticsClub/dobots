import cv2
import numpy as np
import keras
from helpers import show_image
from PIL import Image
#import pytesseract


"""# Load model and save it in cache
def init():
    global model
    model = keras.models.load_model('./assets/ocr_model.h5')"""

import time
def make_prediction(img, neural_network=False, model=None):
    # Preprocess image
    img = np.asarray(img)

    if neural_network:
        img = cv2.resize(img, (64, 64))
        # Crop image borders
        img = img[10:img.shape[0] - 10, 10:img.shape[1] - 10]
        img = cv2.resize(img, (64, 64))
        img = img / 255 # Normalize it
        img = img.reshape(1, 64, 64, 1)
        # Make prediction
        pred = model.predict(img, verbose=0)
        pred_number = np.argmax(pred, axis=1)[0]
        pred_probability = np.amax(pred)
        #print(pred_number, pred_probability)
        
        # Return prediction if probability is high enough
        # Otherwise return 0 (empty cell)
        if pred_probability > 0.65:
            return pred_number
        else:
            return 0
    else:
        print("Tesseract OCR Deprecated")
        img = Image.fromarray(img)
        #img.convert("L").show()
        text = ""#pytesseract.image_to_string(img.convert("L"), config="outputbase digits")
        #print(text)
        #time.sleep(1)
        if text == '' or text == ' ' or text == '  ' or text == '.':
            return 0
        else:
            return int(text)


def get_board_numbers(img, cells, neural_network=False):
    board = []
    model = None
    if neural_network:
        model = keras.models.load_model('./assets/ocr_model.h5')
    for i in cells:
        # Crop cell
        cell = img[int(i[0][1]):int(i[1][1]), int(i[0][0]):int(i[1][0])]
        # Make prediction
        pred = make_prediction(cell, neural_network, model)
        board.append(pred)
    return board


"""if __name__ == '__main__':
    init()"""