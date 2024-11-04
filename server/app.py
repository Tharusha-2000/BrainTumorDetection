# # use fast api taking string as parameter from a get function and converting it to image and then to numpy array and then to a dataframe and then to a prediction and then to a json file and then to a string and then to a response
# from flask import Flask, json, request
# from tensorflow.keras.models import model_from_json
# from flask_cors import CORS, cross_origin
# import numpy as np
# import pandas as pd
# import cv2
# import pickle
# import base64
# from io import BytesIO
# from PIL import Image
# from typing import List
# from pydantic import BaseModel
# import tensorflow as tf

# app = Flask(__name__)
# cors = CORS(app)
# app.config['CORS_HEADERS'] = 'Content-Type'


# # Load model architecture from JSON file
# json_file = open('server/BrainTumor10EpochsCategorical.json', 'r')
# loaded_model_json = json_file.read()
# json_file.close()
# loaded_model = model_from_json(loaded_model_json)

# # Load weights into the model
# loaded_model.load_weights("server/BrainTumor10EpochsCategorical.h5")


# def get_cv2_image_from_base64_string(b64str):
#     encoded_data = b64str.split(',')[1]
#     nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
#     img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
#     return img


# def get_image_from_base64_string(b64str):
#     encoded_data = b64str.split(',')[1]
#     image_data = BytesIO(base64.b64decode(encoded_data))
#     img = Image.open(image_data)
#     return img

# @app.route('/home',methods=['GET'])
# def home():
#     return "Hello World"

# @app.route("/", methods=['POST'])
# def read_root():
#     data = json.loads(request.data)
#     predict_img = []
#     for item in data['image']:
#         #Decode the base64-encoded image
#         image = get_cv2_image_from_base64_string(item)
#         image = cv2.resize(image,(224,224))
#         predict_img.append(image)
#         # encoded_data = item.split(',')[1]
#         # image_data = BytesIO(base64.b64decode(encoded_data))
#         # pil_image = Image.open(image_data)
#         # # Resize the image to 224x224
#         # resized_image = pil_image.resize((224, 224))
#         # # Append the resized image to the list
#         # predict_img.append(resized_image)

#     # np_images = np.array([np.array(img) for img in predict_img])
#     # # Convert the NumPy array to a TensorFlow tensor
#     # tf_images = tf.convert_to_tensor(np_images, dtype=tf.float32)
#     # # # Convert the image to a numpy array
#     prediction = loaded_model.predict(np.array(predict_img))
#     result = np.argmax(prediction, axis=1)

#     # make the probablity frtom prediction
#     # print(prediction[:,1])
#     # print(result)

#     return {"result": prediction[:, 1].tolist()}


# if __name__ == '__main__':
#     app.run(port=5000)


# from flask import Flask, request, jsonify
# from keras.models import load_model

# app = Flask(__name__)

# model = load_model('server/BrainTumor10EpochsCategorical.h5')



# @app.route('/home',methods=['GET'])
# def home():
#     return "Hello World"

# @app.route('/predict', methods=['POST'])
# def predict():
#     data = request.json
#     prediction = model.predict([data])
#     return jsonify(prediction.tolist())

# if __name__ == '__main__':
#     app.run(port=5000)


import os
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from PIL import Image
import cv2
from keras.models import load_model
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename


app = Flask(__name__)


model =load_model('server/BrainTumor10EpochsCategorical.h5')
print('Model loaded. Check http://127.0.0.1:5000/')


def get_className(classNo):
	if classNo==0:
		return "No Brain Tumor"
	elif classNo==1:
		return "Yes Brain Tumor"


def getResult(img):
    image=cv2.imread(img)
    image = Image.fromarray(image, 'RGB')
    image = image.resize((64, 64))
    image=np.array(image)
    image = image / 255.0
    input_img = np.expand_dims(image, axis=0)
    result=model.predict(input_img)
   
    if result.shape[1] > 1:
        predicted_class = np.argmax(result)
        print(f"{predicted_class}")
    else:
        print(f"Predicted value: {result[0][0]}") 
    print(predicted_class)
    return predicted_class


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        f = request.files['file']

        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        value=getResult(file_path)
        predicted_class=get_className(value) 
        return predicted_class
    return None


if __name__ == '__main__':
    app.run(debug=True)    