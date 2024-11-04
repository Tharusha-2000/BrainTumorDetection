import cv2
from keras.models import load_model
from PIL import Image
import numpy as np

model=load_model('server/BrainTumor10EpochsCategorical.h5')

image=cv2.imread('//Users//tharushadinuth//Desktop//brain thomor//BrainTumorDetection//BrainTumorDetection//server//pred//pred5.jpg')

img=Image.fromarray(image)

img=img.resize((64,64))

img=np.array(img)


 # Normalize the image data to [0, 1] range
img = img / 255.0

input_img=np.expand_dims(img, axis=0)

result=model.predict(input_img)

# print(f"{result}")

if result.shape[1] > 1:
        predicted_class = np.argmax(result)
        print(f"{predicted_class}")
else:
        print(f"Predicted value: {result[0][0]}")

        