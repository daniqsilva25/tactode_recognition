# -*- coding: utf-8 -*-
"""mobilenetV2_tactode.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1pqCjnh5IDgxEH_2CiKLT9977xJRnfD15

# MobileNetV2 for Tactode

## Setup
"""

import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import PIL.Image
import tensorflow as tf
import datetime

from tensorflow.keras.preprocessing import image_dataset_from_directory

print("Using TensorFlow version", tf.__version__)

device_name = tf.test.gpu_device_name()
if device_name != '/device:GPU:0':
  raise SystemError('GPU device not found')
print('Found GPU at: {}'.format(device_name))

# Commented out IPython magic to ensure Python compatibility.
from os.path import join
from google.colab import drive

ROOT = "/gdrive"
drive.mount(ROOT)

# Go to directory for model
# %cd
# %cd "/gdrive/My Drive/mobilenetV2_tactode"
# %ls

TACTODE_DATA = "../tactode_data" # path to datasets folder

import pathlib

data_dir = os.path.join(TACTODE_DATA, "imgs_224x224")

data_dir = pathlib.Path(data_dir)

img_count = len(list(data_dir.glob("*/*.jpg")))
print("Total amount of images:", img_count)

"""## Create the dataset"""

BATCH_SIZE = 32
IMG_SIZE = (224, 224)

# Split data for training and validation
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir,
  validation_split=.2,
  subset="training",
  seed=123,
  image_size=IMG_SIZE,
  shuffle=True,
  batch_size=BATCH_SIZE
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir,
  validation_split=.2,
  subset="validation",
  seed=123,
  shuffle=True,
  image_size=IMG_SIZE,
  batch_size=BATCH_SIZE
)

# Show class names
class_names = train_ds.class_names
print("Class names:", class_names)

# Save classes to file
filename = "classes.txt"
file_content = ""

for name in class_names:
  file_content += "%s\n" % name

with open(filename, "w") as f:
  f.write(file_content)

# Update main folder
if not os.path.exists("logs"):
  os.mkdir("logs")
  os.mkdir("logs/model")
  os.mkdir("logs/finetuned")
  print("Created folder 'logs'")

if not os.path.exists("export"):
  os.mkdir("export")
  print("Created folder 'export'")

if not os.path.exists("training"):
  os.mkdir("training")
  os.mkdir("training/model")
  os.mkdir("training/finetuned")
  print("Created folder 'training'")

"""### Visualize the data"""

plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
  for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(images[i].numpy().astype("uint8"))
    plt.title(class_names[labels[i]])
    plt.axis("off")

# Batches from the train dataset
for images_batch, labels_batch in train_ds:
  print(images_batch.shape)
  print(labels_batch.shape)
  break

val_batches = tf.data.experimental.cardinality(val_ds)
test_ds = val_ds.take(val_batches // 5)
val_ds = val_ds.skip(val_batches // 5)

print('Number of validation batches: %d' % tf.data.experimental.cardinality(val_ds))
print('Number of test batches: %d' % tf.data.experimental.cardinality(test_ds))

"""### Configure the dataset for performance"""

AUTOTUNE = tf.data.experimental.AUTOTUNE

train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

"""## Create the model base from pre-trained convnets"""

# Create the base model from the pre-trained model VGG16
IMG_SHAPE = IMG_SIZE + (3,)
base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                               include_top=False,
                                               weights='imagenet')

# Use model method to rescale the images
preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input

# Testing with an example batch of images
image_batch, label_batch = next(iter(train_ds))
feature_batch = base_model(image_batch)
print(feature_batch.shape)

"""## Feature extraction

### Freeze the convolutional base
"""

# Freezing entire model to stop it from update its weights
base_model.trainable = False

# Base model architecture
base_model.summary()

"""### Add a classification head"""

layer_flatten = tf.keras.layers.Flatten()
feature_batch_flatten = layer_flatten(feature_batch)
print(feature_batch_flatten.shape)

layer_dense = tf.keras.layers.Dense(256, activation="relu")
feature_batch_dense = layer_dense(feature_batch_flatten)
print(feature_batch_dense.shape)

prediction_layer = tf.keras.layers.Dense(len(class_names), activation="softmax")
prediction_batch = prediction_layer(feature_batch_dense)
print(prediction_batch.shape)

# Chained model
inputs = tf.keras.Input(shape=IMG_SHAPE)
x = preprocess_input(inputs)
x = base_model(x, training=False) # training=False, cuz model have BatchNorma.
x = layer_flatten(x)
x = layer_dense(x)
x = tf.keras.layers.Dropout(0.5)(x)
outputs = prediction_layer(x)
model = tf.keras.Model(inputs, outputs)

"""### Compile the model"""

base_learning_rate = 0.0001

model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=base_learning_rate),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['accuracy'])
model.summary()

len(model.trainable_variables)

"""### Train the model"""

initial_epochs = 20

loss0, accuracy0 = model.evaluate(val_ds)

print("initial loss: {:.2f}".format(loss0))
print("initial accuracy: {:.2f}".format(accuracy0))

# Commented out IPython magic to ensure Python compatibility.
"""
# %load_ext tensorboard

if os.path.exists("logs/model/*"):
#   %rm -r logs/model/*

log_dir = os.path.join("logs/model",
                      datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir, histogram_freq=1)

# %tensorboard --logdir /logs/model/
"""
#model.load_weights("training/model/2020.../cp.cpkt")

ckptdir = os.path.join("training/model",
                       datetime.datetime.now().strftime("%Y%m%d-%H%M%S"),
                       "cp.ckpt")
ckpt_callback = tf.keras.callbacks.ModelCheckpoint(filepath=ckptdir,
                                                 save_weights_only=True,
                                                 verbose=1)

history = model.fit(train_ds,
                    epochs=initial_epochs,
                    validation_data=val_ds,
                    callbacks=[ckpt_callback])

"""### Show curves"""

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
plt.ylim([min(plt.ylim()),1.0])
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.ylabel('Cross Entropy')
plt.ylim([0,1.0])
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.show()

# Uncomment below if needed
#model.load_weights("training/model/20200926-151817/cp.ckpt")

# Commented out IPython magic to ensure Python compatibility.
# Save the model
if os.path.exists("export/model"):
#   %rm -r export/m*
model.save("export/model")

# Check model dir content
!ls export/model

"""## Fine tuning

### Unfreeze top layers of the model
"""

base_model.trainable = True
# Number of layers composing the base model
print("Number of layers in the base model: ", len(base_model.layers))

# Number of top layers to unfreeze (finetune)
num_layers_to_finetune = 47

# Fine-tune from this layer onwards
fine_tune_at = len(base_model.layers) - num_layers_to_finetune

# Freeze all the layers before the `fine_tune_at` layer
for layer in base_model.layers[:fine_tune_at]:
  layer.trainable =  False

# Show layers to be finetuned
for layer in base_model.layers:
  print("L: " + str(layer) + " - Trainable: " + str(layer.trainable))

"""### Compile the model"""

model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              optimizer = tf.keras.optimizers.RMSprop(lr=base_learning_rate/10),
              metrics=['accuracy'])
model.summary()

len(model.trainable_variables)

"""### Continue traning the model"""

# Commented out IPython magic to ensure Python compatibility.
"""
# %reload_ext tensorboard

if os.path.exists("logs/finetuned/*"):
#   %rm -r logs/finetuned/*

log_dir = os.path.join("logs/finetuned",
                      datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir, histogram_freq=1)

# %tensorboard --logdir /logs/finetuned/
"""
#model.load_weights("training/model/2020.../cp.cpkt")

ckptdir = os.path.join("training/finetuned",
                       datetime.datetime.now().strftime("%Y%m%d-%H%M%S"),
                       "cp.ckpt")
ckpt_callback = tf.keras.callbacks.ModelCheckpoint(filepath=ckptdir,
                                                 save_weights_only=True,
                                                 verbose=1)

fine_tune_epochs = 20
total_epochs =  initial_epochs + fine_tune_epochs

history_fine = model.fit(train_ds,
                         epochs=total_epochs,
                         initial_epoch=history.epoch[-1],
                         validation_data=val_ds,
                         callbacks=[ckpt_callback])

fine_acc = history_fine.history['accuracy']
fine_val_acc = history_fine.history['val_accuracy']

fine_loss = history_fine.history['loss']
fine_val_loss = history_fine.history['val_loss']

global_acc = acc + fine_acc
global_val_acc = val_acc + fine_val_acc

global_loss = loss + fine_loss
global_val_loss = val_loss + fine_val_loss

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(global_acc, label='Training Accuracy')
plt.plot(global_val_acc, label='Validation Accuracy')
plt.ylim([0.5, 1.05])
plt.xlim([0, total_epochs + 1])
plt.plot([initial_epochs-1,initial_epochs-1],
          plt.ylim(), label='Start Fine Tuning')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(global_loss, label='Training Loss')
plt.plot(global_val_loss, label='Validation Loss')
plt.ylim([-0.05, 1.0])
plt.xlim([0, total_epochs + 1])
plt.plot([initial_epochs-1,initial_epochs-1],
         plt.ylim(), label='Start Fine Tuning')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.show()

"""### Evaluation and prediction"""

# Uncomment below if needed
#latest = tf.train.latest_checkpoint("training/finetuned/20200924-170325")
#print("Latest checkpoint:", latest)
#model.load_weights(latest)

loss, accuracy = model.evaluate(test_ds)
print('Test accuracy :', accuracy)

# Commented out IPython magic to ensure Python compatibility.
# Save the finetuned model
if os.path.exists("export/finetuned"):
#   %rm -r export/f*
model.save("export/finetuned")

# Check finetuned model dir content
!ls export/finetuned