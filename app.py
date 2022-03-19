import os
from flask import Flask, render_template, request, redirect, send_file, url_for, jsonify, flash
from backend.helpers import upload_file_to_s3
from urllib.request import urlretrieve
import joblib
from datetime import datetime
from io import BytesIO

from PIL import Image
import cv2
import numpy as np

app = Flask(__name__)
app.config['SECRET KEY'] = 'a secret key'
app.secret_key = 'super secret key'

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
BUCKET_NAME = os.getenv("AWS_BUCKET_NAME")
UPLOAD_URL = "https://oralsnap-bucket.s3.ca-central-1.amazonaws.com/"

@app.route("/")
def home():
    return render_template("index.html")

# Check file extension
def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def detect_blur(image, threshold=90):
    laplacian_var = cv2.Laplacian(image, cv2.CV_64F).var()
    if laplacian_var > threshold:
        return True
    return False

@app.route("/upload", methods=["POST"])
def create():
    # isBlurry = False
    if request.method == 'POST':
        file = request.files['file']

        # Check whether the file extension is allowed (eg. png,jpeg,jpg)
        if file and allowed_file(file.filename):
            dateTimeObj = datetime.now()
            # Rename file to contain timestamp
            if (file.filename).rsplit('.', 1)[1].lower() in ['png', 'jpg']:
                name = file.filename[:-4] + "_" + str(dateTimeObj.strftime("%Y-%m-%d-%H-%M-%S")) + ".png"
            else:
                name = file.filename[:-5] + "_" + str(dateTimeObj.strftime("%Y-%m-%d-%H-%M-%S")) + ".png"
            
            # Upload image to s3 bucket
            output = upload_file_to_s3(file, filename=name)
            img = UPLOAD_URL + output
            return redirect(url_for('analysis', URL=img))
        else:
            flash("Please try again.")
            return redirect(url_for('home'))

@app.route('/predict')
def analysis():
    decayCount = 0
    isDecay = False
    imagePath = request.args.get('URL')[len(UPLOAD_URL):]
    urlretrieve(request.args.get('URL'), imagePath)

    img = Image.open(imagePath)
    img = img.convert('RGB')

    # Image Segmentaion - Otsu's Method
    img_cv = cv2.imread(imagePath)
    img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    image_gray = cv2.cvtColor(img_cv, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(image_gray, (11, 11), 0) 
    _, thresh = cv2.threshold(blurred, 180, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    masked = Image.fromarray(cv2.bitwise_and(img_cv, img_cv, mask=thresh))

    width, height = masked.size
    image = np.zeros((width, height, 3))
    colors = []

    # Get all colours in segmented image
    for x in range(0, width):
        for y in range(0, height):
            if masked.getpixel((x,y))[0] == 0 and masked.getpixel((x,y))[1] == 0 and masked.getpixel((x,y))[2] == 0:
                masked.putpixel( (x, y), (255, 255, 255, 255) )
            colors.append( masked.getpixel((x,y)) )

    colors = np.array(colors)
    # Load model
    model = joblib.load('backend/trained_model.joblib')
    predictor = np.array( model.predict(colors) ).reshape(width, height)

    for x in range(0, width):
        for y in range(0, height):
            if predictor[x][y] == 0:
                # RED - Caries
                decayCount += 1
                image[x][y] = [255 , 0, 0]
            else:
                # WHITE - Healthy Teeth or Gums
                image[x][y] = [255, 255, 255]

    # Overlay Red Markings on Original Image
    image = np.fliplr ( np.rot90(image, k=1, axes=(1,0)) )
    image = np.array(image, dtype=np.uint8)
    image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    _, mask = cv2.threshold(image_gray, 200, 255, cv2.THRESH_BINARY_INV)
    mask_inv = cv2.bitwise_not(mask)
    img1_bg = cv2.bitwise_and(img_cv, img_cv, mask = mask_inv)
    img2_fg = cv2.bitwise_and(image, image, mask = mask)
    result = Image.fromarray( cv2.add(img1_bg,img2_fg) )
    # Save Prediction
    outputPath = "output_" + imagePath
    result.save(outputPath)
    with open(outputPath, 'rb') as data:
        output_image = upload_file_to_s3(data, filename=outputPath)
    
    if decayCount > 0:
        isDecay = True
    response = {
        "decay": isDecay, # True or False
        "imageURL": UPLOAD_URL + output_image # string: URL
    }
    # Uncomment to get json response
    return jsonify(response)
    # Uncomment to render image on page
    # return render_template('response.html', output_image = UPLOAD_URL + output_image)


if __name__ == '__main__':
    app.run(debug=True)