import numpy as np
import tensorflow as tf

rep_ds = tf.data.Dataset.list_files("./quant-images/*.jpg")
HEIGHT, WIDTH = 640, 640

def representative_dataset_gen():
   for image_path in rep_ds:
       img = tf.io.read_file(image_path)
       img = tf.io.decode_image(img, channels=3)
       img = tf.image.convert_image_dtype(img, tf.float32)
       resized_img = tf.image.resize(img, (HEIGHT, WIDTH))
       resized_img = resized_img[tf.newaxis, :]
       yield [resized_img]

model_path = './exported-models/ptag-detector-model/ssd_mobilenet_v2_fpnlite_640x640/saved_model'        

#import trained model from mobilenet 640 v2 fpn
converter = tf.lite.TFLiteConverter.from_saved_model(model_path) #using tensorflow

converter.representative_dataset = representative_dataset_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8
converter.quantized_input_stats = {"normalized_input_image_tensor": (128, 128)}
converter.optimizations = [tf.lite.Optimize.DEFAULT]

tflite_model_quant = converter.convert()

interpreter = tf.lite.Interpreter(model_content=tflite_model_quant)
input_type = interpreter.get_input_details()[0]['dtype']
print('input: ', input_type)
output_type = interpreter.get_output_details()[0]['dtype']
print('output: ', output_type)

# Save the model.
with open('detect_quant.tflite', 'wb') as f:
  f.write(tflite_model_quant)
