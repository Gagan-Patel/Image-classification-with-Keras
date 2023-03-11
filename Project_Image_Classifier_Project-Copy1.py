#!/usr/bin/env python
# coding: utf-8

# ## Install Datasets and Upgrade TensorFlow
# 
# download the latest version of the `oxford_flowers102` dataset, let's first install both `tensorflow-datasets` and `tfds-nightly`.
# upgrade TensorFlow to ensure we have a version that is compatible with the latest version of the dataset.

# In[1]:


get_ipython().run_line_magic('pip', '--no-cache-dir install tensorflow-datasets --user')
get_ipython().run_line_magic('pip', '--no-cache-dir install tfds-nightly --user')
get_ipython().run_line_magic('pip', '--no-cache-dir install --upgrade tensorflow --user')


# After the above installations have finished **be sure to restart the kernel**.

# In[1]:


# Import TensorFlow 
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub

# Ignore some warnings that are not relevant (you can remove this if you prefer)
import warnings
warnings.filterwarnings('ignore')


# In[2]:


# Make all other necessary imports.
import time
import numpy as np
import matplotlib.pyplot as plt

import json
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import logging
logger = tf.get_logger()
logger.setLevel(logging.ERROR)


# In[3]:


# Some other recommended settings:
get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")
tfds.disable_progress_bar()


# In[4]:


print('Using:')
print('\t\u2022 TensorFlow version:', tf.__version__)
print('\t\u2022 tf.keras version:', tf.keras.__version__)
print('\t\u2022 Running on GPU' if tf.test.is_gpu_available() else '\t\u2022 GPU device not found. Running on CPU')


# ## Load the Dataset
# 
# Here you'll use `tensorflow_datasets` to load the [Oxford Flowers 102 dataset](https://www.tensorflow.org/datasets/catalog/oxford_flowers102). This dataset has 3 splits: `'train'`, `'test'`, and `'validation'`.  You'll also need to make sure the training data is normalized and resized to 224x224 pixels as required by the pre-trained networks.
# 
# The validation and testing sets are used to measure the model's performance on data it hasn't seen yet, but you'll still need to normalize and resize the images to the appropriate size.

# In[5]:


#  the dataset with TensorFlow Datasets. Hint: use tfds.load()

dataset, dataset_info = tfds.load('oxford_flowers102', as_supervised=True, with_info=True)

# Create a training set, a validation set and a test set.

train_set, val_set, test_set = dataset['train'],dataset['validation'], dataset['test']


# ## Explore the Dataset

# In[6]:


#  Get the number of examples in each set from the dataset info.
num_training = dataset_info.splits['train'].num_examples
num_validation = dataset_info.splits['validation'].num_examples
num_test = dataset_info.splits['test'].num_examples

total_num_examples = dataset_info.splits['train'].num_examples

print('The Dataset has a total of:')
print('\u2022{:,} examples'.format(total_num_examples))
print('\nThere are {:,} images in the test set.'.format(num_test))
print('There are {:,} images in the validation set.'.format(num_validation))
print('There are {:,} images in the training set.\n'.format(num_training))

# Get the number of classes in the dataset from the dataset info.

num_classes = dataset_info.features['label'].num_classes

print("Number of the Classes:", num_classes)


# In[7]:


# Print the shape and corresponding label of 3 images in the training set.
  
for image, label in train_set.take(3):
    image = image.numpy()
    label = label.numpy()

    plt.imshow(image)
    plt.show()

    print('The shape of the image:', image.shape)
    print('The label of the image:', label)


# In[8]:


# Plot 1 image from the training set. 
for image, label in train_set.take(1):
    image = image.numpy()
    label = label.numpy()

plt.imshow(image)
# Set the title of the plot to the corresponding image label. 
plt.title(label)


# ### Label Mapping
# Dictionary mapping the integer coded labels to the actual names of the flowers can be loaded in a mapping of `label_map.json`.

# In[9]:


with open('label_map.json', 'r') as f:
    class_names = json.load(f)


# In[10]:


# Plot 1 image from the training set. Set the title of the plot to the corresponding class name. 
for image, label in train_set.take(1):
    image = image.numpy()
    label = label.numpy()

plt.imshow(image)
plt.title(class_names[str(label)])


# ## Create Pipeline

# In[11]:


# Create a pipeline for each set.

batch_size = 32
image_size = 224

def format_image(image, label):
    image = tf.cast(image, tf.float32)
    image = tf.image.resize(image, (image_size, image_size))
    image /= 255
    return image, label

training_batches = train_set.shuffle(num_training//4).map(format_image).batch(batch_size).prefetch(1)
validation_batches = val_set.map(format_image).batch(batch_size).prefetch(1)
testing_batches = test_set.map(format_image).batch(batch_size).prefetch(1)


# # Build and Train the Classifier

# In[12]:


# Load the MobileNet pre-trained network from TensorFlow Hub
URL = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4"

feature_extractor = hub.KerasLayer(URL, input_shape=(224, 224,3))
feature_extractor.trainable = False


# In[13]:



# Build and train your network.
model = tf.keras.Sequential([
        feature_extractor,
        tf.keras.layers.Dense(num_classes, activation = 'softmax')
])

model.summary()
print('Is there a GPU Available:\n', tf.test.is_gpu_available())

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

EPOCHS = 5

history = model.fit(training_batches,
                    epochs=EPOCHS,
                    validation_data=validation_batches)


# In[14]:


# Plot the loss and accuracy values achieved during training for the training and validation set.

training_accuracy = history.history['accuracy']
validation_accuracy = history.history['val_accuracy']

training_loss = history.history['loss']
validation_loss = history.history['val_loss']

epochs_range=range(EPOCHS)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, training_accuracy, label='Training Accuracy')
plt.plot(epochs_range, validation_accuracy, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, training_loss, label='Training Loss')
plt.plot(epochs_range, validation_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()


# ## Testing your Network

# In[15]:


# TODO: Print the loss and accuracy values achieved on the entire test set.
for image_batch, label_batch in training_batches.take(1):
    loss, accuracy = model.evaluate(image_batch, label_batch)

    print(f"The accuracy of trained model is: {accuracy}")
    print(f"The loss of trained model is: {loss}")


# ## Save the Model

# In[16]:


# Save your trained model as a Keras model.

t= time.time()
saved_keras_model = './Keras_model.h5'.format(int(t))
model.save(saved_keras_model)


# ## Load the Keras Model

# In[17]:


#  Load the Keras model
reloaded_keras_model = tf.keras.models.load_model(saved_keras_model, 
                                                  custom_objects = {'KerasLayer':hub.KerasLayer})

reloaded_keras_model.summary()


# # Inference for Classification

# In[18]:


# Create the process_image function
def process_image(image):
    image = image.squeeze()
    image = tf.image.resize(image, [image_size, image_size])
    image /= 255
    image = image.numpy()
    return image


# To check your `process_image` function we have provided 4 images in the `./test_images/` folder:
# 
# * cautleya_spicata.jpg
# * hard-leaved_pocket_orchid.jpg
# * orange_dahlia.jpg
# * wild_pansy.jpg
# 
# The code below loads one of the above images using `PIL` and plots the original image alongside the image produced by your `process_image` function. If your `process_image` function works, the plotted image should be the correct size. 

# In[19]:


from PIL import Image

image_path = './test_images/hard-leaved_pocket_orchid.jpg'
im = Image.open(image_path)
test_image = np.asarray(im)

processed_test_image = process_image(test_image)

fig, (ax1, ax2) = plt.subplots(figsize=(10,10), ncols=2)
ax1.imshow(test_image)
ax1.set_title('Original Image')
ax2.imshow(processed_test_image)
ax2.set_title('Processed Image')
plt.tight_layout()
plt.show()


# ### Inference

# In[20]:


#  Create the predict function
def predict(image_path, model, top_k):
    image = Image.open(image_path)
    image = np.asarray(image)
    processed_image = process_image(image)
    expanded_image = np.expand_dims(processed_image, axis=0)
    
    predictions = model.predict(expanded_image)
    probs, labels = tf.nn.top_k(predictions, k=top_k)
    probs = list(probs.numpy()[0])
    labels = list(labels.numpy()[0])


    return probs, labels, processed_image


# In[21]:


image_path = './test_images/hard-leaved_pocket_orchid.jpg'
image = np.asarray(Image.open(image_path))

probs = predict(image_path, reloaded_keras_model, 5)

print(probs)


# In[22]:


image_path = './test_images/hard-leaved_pocket_orchid.jpg'
image = np.asarray(Image.open(image_path))
classes = predict(image_path, reloaded_keras_model, 5)

print(classes)


# In[23]:


predict('./test_images/cautleya_spicata.jpg', reloaded_keras_model, 5 )


# # Sanity Check

# #  Plot the input image along with the top 5 classes    

# In[40]:


# the glob module is used to retrieve files/pathnames matching a specified pattern.
import glob
top_k = 5

for image_path in glob.glob('./test_images/*.jpg'):
    flower_name = image_path[14:-4].title().replace('_', ' ')
    probs, labels, processed_image = predict(image_path, reloaded_keras_model, top_k)

    top_k_class_names = []
    for i in labels:
        top_k_class_names.append(class_names[str(i+1)].title())

    fig, (ax1, ax2) = plt.subplots(figsize=(10,9), ncols=2)
    ax1.imshow(processed_image)
    ax1.axis('off')
    ax1.set_title('{}'.format(flower_name))
    ax2.barh(np.arange(top_k), probs)
    ax2.set_aspect(0.1)
    ax2.set_yticks(np.arange(top_k))
    ax2.set_yticklabels(top_k_class_names, size='small');
    ax2.set_title('Class Probability')
    ax2.set_xlim(0, 1.1)
    plt.tight_layout()


# In[ ]:




