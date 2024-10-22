from flask import *
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import cv
from tensorflow.keras.preprocessing.image import load_img, img_to_array, array_to_img
import pyttsx3
import pickle
import os
import subprocess


app = Flask(__name__)

model = load_model('new_sign_language_model.h5')


map_characters = {0: '1', 1: '2', 2: '3', 3: '4', 4: '5', 5: '6', 6: '7', 7: '8', 8: '9', 9: 'A', 10: 'B', 11: 'C', 12: 'D',
                  13: 'E', 14: 'F', 15: 'G', 16: 'H', 17: 'I', 18: 'J', 19: 'K', 20: 'L', 21: 'M', 22: 'N', 23: 'O', 24: 'P',
                  25: 'Q', 26: 'R', 27: 'S', 28: 'T', 29: 'U', 30: 'V', 31: 'W', 32: 'X', 33: 'Y', 34: 'Z'}

@app.route("/")
@app.route("/index")
def index():
	return render_template('index.html')

@app.route("/login")
def login():
	return render_template('login.html')

@app.route('/main')
def main():
    return render_template("main.html")

@app.route('/predict',methods=['post'])
def predict():
    try:
        img = request.files['img']

        img_path = "static/tests/" + img.filename   
        print(img_path)
        img.save(img_path)

        class_labels = list(map_characters.values())
        print(class_labels)

        img_read = cv2.imread(img_path,0)
        print(img_read.shape)
        img_read = edge_detection(img_read)
        img_read = cv2.resize(img_read, (64, 64))
        print(img_read.shape)
        img_read = img_to_array(img_read)

        p = img_read.reshape(1, 64, 64, 1)

        # Predict the label
        prediction = model.predict(p)
        print(prediction)
        predicted_index = np.argmax(prediction)
        print(predicted_index)

        if predicted_index is not None and predicted_index < len(class_labels):
            predicted_label = class_labels[predicted_index]
            print(predicted_label)

            engine = pyttsx3.init()  # Initialize the text-to-speech engine
            engine.say(predicted_label)
            engine.runAndWait()  # Run and wait for speech completion

            return render_template('main.html', value=predicted_label)
        else:
            print("Invalid prediction index.")
            return render_template('main.html', value="Error in prediction")
    except Exception as e:
        print("Error: ", str(e))
        return render_template('main.html', value="Error during prediction")



@app.route('/real_main')
def real_main():
    # Load the trained model
    try:
        # Execute the script with subprocess
        subprocess.run(["python", "yolov5/detect.py", "--weights", "best.pt", "--img", "416", "--conf", "0.5", "--source", "0"])
        return render_template('main.html')
    except Exception as e:
        # Handle potential errors during script execution (optional)
        print(f"Error running detect.py: {e}")
        return "An error occurred. Please check the server logs for details."
    
  
def edge_detection(image):
    minValue = 70
    blur = cv2.GaussianBlur(image,(5,5),2)
    th3 = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,11,2)
    ret, res = cv2.threshold(th3, minValue, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    return res

if __name__ == '__main__':
    app.run(debug=True)