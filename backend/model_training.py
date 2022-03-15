# Import Libraries
from PIL import Image, ImageColor
import cv2
import numpy as np
import pandas as pd
from joblib import dump, load
from matplotlib import pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
from sklearn import svm

# Data Pre-processing
caries_colors = pd.read_csv('caries_colours.csv', header=None)
caries_colors = caries_colors[0].to_list()
decay_colors = []
for c in caries_colors:
  new = ImageColor.getcolor(c,"RGB")
  decay_colors.append(new)
caries = np.array(decay_colors)

healthy_colors = pd.read_csv('healthy_colours.csv', header=None)
healthy_colors = healthy_colors[0].to_list()
healthy = []
for c in healthy_colors:
  new = ImageColor.getcolor(c,"RGB")
  healthy.append(new)
healthy = np.array(healthy)

gum_colors = pd.read_csv('gum_colours.csv', header=None)
gum_colors = gum_colors[0].to_list()
gum = []
for c in gum_colors:
  new = ImageColor.getcolor(c, "RGB")
  gum.append(new)
gum = np.array(gum)

# Data - X
X = np.concatenate((caries, healthy, gum), axis=0)

# Labels - y
zeros = np.zeros(130) # Caries
ones = np.ones(100) # Healthy 
twos = 2 * np.ones(125) # Gum
y = np.concatenate((zeros, ones, twos))

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)
class_names = ['Caries','Healthy','Gum']

# Model Training
model = svm.SVC(kernel='linear', C=1, class_weight={0: 2.5, 1: 2, 2:4.8}).fit(X_train, y_train)
y_pred_lin = model.predict(X_test)

print(classification_report(y_test, y_pred_lin, target_names=class_names))

# Save model
dump(model, 'trained_model.joblib')