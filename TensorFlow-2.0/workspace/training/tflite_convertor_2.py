import tensorflow as tf
import numpy as np
import cv2
import glob

model_path = './exported-models/ptag-detector-model/ssd_mobilenet_v2_fpnlite_320x320/saved_model'

def representative_data_gen():
  dataset_list = tf.data.Dataset.list_files('./quant-images/*.jpg')
  for i in range(100):
    image = next(iter(dataset_list))
    image = tf.io.read_file(image)
    image = tf.io.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [320, 320])
    image = image[np.newaxis,:,:,:]
    image = image - 127.5
    image = image * 0.007843
    yield [image]

converter = tf.lite.TFLiteConverter.from_saved_model(model_path)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8, tf.lite.OpsSet.SELECT_TF_OPS]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8
converter.allow_custom_ops = True
converter.representative_dataset = representative_data_gen
tflite_model = converter.convert()
  
with open('detect_quant.tflite', 'wb') as f:
  f.write(tflite_model)
