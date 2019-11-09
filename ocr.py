import cv2
import pytesseract
import os
from PIL import Image
import numpy as np

pytesseract.pytesseract.tesseract_cmd = r"c:\\Program Files\\Tesseract-OCR\\tesseract.exe"


for root, dirs, files in os.walk("data"):
    for file in files:
        img = Image.open(os.path.join(root, file))
        # img = cv2.imread(os.path.join(os.path.join(root, file)))
        # img = np.array(raw_img)
        text = pytesseract.image_to_string(img)
        # print(pytesseract.image_to_data(img))
        print(text)
        print(40*"-")
        img.close()
