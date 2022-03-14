import os
from flask import Flask, render_template, request, redirect, send_file, url_for
from helpers import upload_file_to_s3
from urllib.request import urlretrieve
import joblib

from PIL import Image, ImageColor
import cv2
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

app = Flask(__name__)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
BUCKET_NAME = os.getenv("AWS_BUCKET_NAME")
UPLOAD_URL = "https://s3.amazonaws.com/${BUCKET_NAME}$"

@app.route("/")
def home():
    # return "<p>Sample response from OralSnap app!</p>"
    return render_template("index.html")

# function to check file extension
def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/upload", methods=["POST"])
def create():
    if request.method == 'POST':
        file = request.files['file']
        # check whether the file extension is allowed (eg. png,jpeg,jpg)
        if file and allowed_file(file.filename):
            output = upload_file_to_s3(file)
            return output
            # return redirect(url_for('analysis', URL=output))

        else:
            flash("Please try again.")
            return redirect(url_for('new'))

# @app.route('/predict')
# def analysis():
#     urlretrieve(request.args.get('URL'), "uploaded_image.png")
#     img = Image.open("uploaded_image.png")
#     img = img.convert('RGB')

#     # get only teeth (not gums)
#     img_cv = cv2.imread(imagePath)
#     img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
#     image_gray = cv2.cvtColor(img_cv, cv2.COLOR_RGB2GRAY)
#     blurred = cv2.GaussianBlur(image_gray, (11, 11), 0) 
#     _, thresh = cv2.threshold(blurred, 180, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
#     masked = Image.fromarray(cv2.bitwise_and(img_cv, img_cv, mask=thresh))

#     width, height = masked.size
#     image = np.zeros((width, height, 3))
#     colors = []

#     # get all colors in the teeth
#     for x in range(0, width):
#         for y in range(0, height):
#         if masked.getpixel((x,y))[0] == 0 and masked.getpixel((x,y))[1] == 0 and masked.getpixel((x,y))[2] == 0:
#             masked.putpixel( (x, y), (255, 255, 255, 255) )
#         colors.append( masked.getpixel((x,y)) )

#     colors = np.array(colors)
#     predictor = np.array( clf.predict(colors) ).reshape(width, height) # SVM prediction

#     for x in range(0, width):
#         for y in range(0, height):
#             if predictor[x][y] == 0:
#             # RED - Caries
#             image[x][y] = [255 , 0, 0]
#             else:
#             # WHITE - Healthy Teeth or Gums
#             image[x][y] = [255, 255, 255]
    
#     image = np.fliplr ( np.rot90(image, k=1, axes=(1,0)) )
#     image = np.array(image, dtype=np.uint8)
#     image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
#     _, mask = cv2.threshold(image_gray, 200, 255, cv2.THRESH_BINARY_INV)
#     mask_inv = cv2.bitwise_not(mask)
#     img1_bg = cv2.bitwise_and(img_cv, img_cv, mask = mask_inv)
#     img2_fg = cv2.bitwise_and(image, image, mask = mask)
#     result = cv2.add(img1_bg,img2_fg)

#     output_image = upload_file_to_s3(result)

#     return output_image


if __name__ == '__main__':
    port = 5000 + random.randint(0, 999)
    model = joblib.load('trained_model.joblib')
    app.run(debug=True, port=port)