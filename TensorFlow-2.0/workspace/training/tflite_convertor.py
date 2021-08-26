import tensorflow as tf
import os
import numpy as np
import cv2

saved_model_dir = './exported-models/ptag-detector-model/ssd_mobilenet_v2_fpnlite_320x320/saved_model'
images_path = './quant-images/'

NORM_H = 320
NORM_W = 320

def representative_dataset_gen():
 for f_name in os.listdir(images_path):
    file_path = os.path.normpath(os.path.join(images_path, f_name))
    img = cv2.imread(file_path)
    img = cv2.resize(img, (NORM_H, NORM_W))
    img = np.reshape(img, (1, NORM_H, NORM_W, 3))
    img = np.asarray(img, dtype='float32')
    img /= 255.0
    yield[img]

converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.allow_custom_ops = True
converter.representative_dataset = representative_dataset_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8  # or tf.int8
converter.inference_output_type = tf.int8  # or tf.int8
converter.experimental_new_converter = True
tflite_model = converter.convert()

# Save the model.
with open('detect.tflite', 'wb') as f:
  f.write(tflite_model)
