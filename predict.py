# Import tensorflow
import tensorflow as tf
import tensorflow_hub as hub
# Import other necessaries
import argparse
import json
import numpy as np
from PIL import Image
image_size = 224
# Step 1: Load model
def load_model(model):
    model = tf.keras.models.load_model(model, custom_objects={'KerasLayer':hub.KerasLayer},compile=False)
    return model
# Step 2: Process image or format image
def process_image(image_path):
    image = np.asarray(Image.open(image_path))
    image = tf.cast(image, tf.float32)
    image = tf.image.resize(image, (image_size, image_size))
    image /= 255
    image = image.numpy()
    return np.expand_dims(image, axis=0)

# Step 3: Prediction using my pretrained model
def predict(image, model, top_k):
    predictions = model.predict(image)
    
    if top_k == None:
        max_class = np.argmax(predictions[0])
        return [predictions[0][max_class]], [max_class]
    
    classes = (-predictions[0]).argsort()[:top_k]
    probabilities = predictions[0][classes]
    return probabilities, classes
parser = argparse.ArgumentParser(
    description='Flower Classification Deep Neural Network with Transfer Learning',
)
parser.add_argument('image_path', action="store", type=str)
parser.add_argument('model', action="store", type=str)
parser.add_argument('--top_k', action="store",
                    dest="top_k", type=int)
parser.add_argument('--category_names', action="store",
                    dest="category_names_json", type=str)
args = parser.parse_args()
# Image processing and Loading the model
processed_image = process_image(args.image_path)
model = load_model(args.model)
# Prediction step
probabilities, classes = predict(processed_image, model, args.top_k)
# Results
print("Class \t\t Probability")
print("====== \t\t ============")
if args.category_names_json == None:
    for i in range(len(probabilities)):
        print(f"{str(classes[i] + 1)}\t\t {probabilities[i]:.3%}")
else:
    with open(args.category_names_json, 'r') as f:
        class_names = json.load(f)
    for i in range(len(probabilities)):
        print(f"{class_names[str(classes[i]+1)]}     \t{probabilities[i]:.3%}")