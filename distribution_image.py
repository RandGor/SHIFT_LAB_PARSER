from pathlib import Path
import pandas as pd
import tensorflow as tf
import numpy as np
import os
from keras.preprocessing.image import ImageDataGenerator

from shutil import copyfile
from pathlib import Path


def data_tf(path_img, BATCH_SIZE = 40):
  AUTOTUNE=tf.data.experimental.AUTOTUNE
  data_dir = Path(path_img)
  list_ds = tf.data.Dataset.list_files(str(data_dir/'*/*'))
  img_ds = list_ds.map(process_data, num_parallel_calls=AUTOTUNE)

  ds = img_ds.batch(BATCH_SIZE)
  ds = ds.prefetch(buffer_size=AUTOTUNE)
  return ds

def decode_img(img):
  img = tf.image.decode_jpeg(img, channels=3)
  img = tf.image.convert_image_dtype(img, tf.float32)
  return tf.image.resize(img, (img_height, img_width))

def process_data(file_path):
    img = tf.io.read_file(file_path)
    img = decode_img(img)
    return img


def predict(model, path_img, img_size):
    global img_height, img_width 
    img_height, img_width = img_size
    data = data_tf(path_img)

    Y_pred = model.predict(data)
    y_pred = np.argmax(Y_pred, axis=1)

    TEST_DIR = Path(path_img)
    test_files = sorted(list(TEST_DIR.rglob('*.*')))

    data_file = pd.DataFrame(test_files)
    data_file['class'] = y_pred
    return data_file


def distribution(classes, data_file):
    for i in classes:
        try:
            os.mkdir(i)
        except FileExistsError:
            pass
    for pathh, classs in data_file.iterrows():
        copyfile(classs[0], Path(str(classs['class']) + '/' + str(pathh) + '.jpg'))