import os
from flask import Flask, render_template, request, redirect, send_file
from s3_functions import upload_file
from werkzeug.utils import secure_filename
from urllib.request import urlretrieve
import joblib

from PIL import Image, ImageColor
import cv2
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
BUCKET = "oralsnap-bucket"

@app.route("/")
def home():
    return "<p>Sample response from OralSnap app!</p>"


@app.route('/predict', methods=['POST'])
def predict(filename):
    urlretrieve("https://s3.amazonaws.com/${BUCKET}$/${UPLOAD_FOLDER}$/${filename}$", "uploaded_image.png")
    img = Image.open("uploaded_image.png")
    img = img.convert('RGB')

    # get only teeth (not gums)
    img_cv = cv2.imread(imagePath)
    img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    image_gray = cv2.cvtColor(img_cv, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(image_gray, (11, 11), 0) 
    _, thresh = cv2.threshold(blurred, 180, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    masked = Image.fromarray(cv2.bitwise_and(img_cv, img_cv, mask=thresh))

    width, height = masked.size
    image = np.zeros((width, height, 3))
    colors = []

    # get all colors in the teeth
    for x in range(0, width):
        for y in range(0, height):
        if masked.getpixel((x,y))[0] == 0 and masked.getpixel((x,y))[1] == 0 and masked.getpixel((x,y))[2] == 0:
            masked.putpixel( (x, y), (255, 255, 255, 255) )
        colors.append( masked.getpixel((x,y)) )

    colors = np.array(colors)
    predictor = np.array( clf.predict(colors) ).reshape(width, height) # SVM prediction

    for x in range(0, width):
        for y in range(0, height):
            if predictor[x][y] == 0:
            # RED - Caries
            image[x][y] = [255 , 0, 0]
            else:
            # WHITE - Healthy Teeth or Gums
            image[x][y] = [255, 255, 255]
    
    image = np.fliplr ( np.rot90(image, k=1, axes=(1,0)) )
    image = np.array(image, dtype=np.uint8)
    image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    _, mask = cv2.threshold(image_gray, 200, 255, cv2.THRESH_BINARY_INV)
    mask_inv = cv2.bitwise_not(mask)
    img1_bg = cv2.bitwise_and(img_cv, img_cv, mask = mask_inv)
    img2_fg = cv2.bitwise_and(image, image, mask = mask)
    result = cv2.add(img1_bg,img2_fg)

    upload(result)

    response = {
    "method": "POST",
    "status": 200
    }

    return jsonify(response)


if __name__ == '__main__':
    model = joblib.load('trained_model.joblib')
    app.run(debug=True)