import tensorflow as tf 
frozen_graph_file = '././exported-models/ptag-detector-model/saved_model/saved_model.pb'
input_arrays = ["normalized_input_image_tensor"] 
output_arrays = ['TFLite_Detection_PostProcess', 
           'TFLite_Detection_PostProcess:1', 
           'TFLite_Detection_PostProcess:2', 
           'TFLite_Detection_PostProcess:3'] 
input_shapes = {"normalized_input_image_tensor" : [1, 320, 320, 3]} 

converter = tf.lite.TFLiteConverter.from_frozen_graph(frozen_graph_file, 
                                               input_arrays=input_arrays, 
                                               output_arrays=output_arrays, 
                                                 input_shapes=input_shapes) 
converter.allow_custom_ops = True 
converter.optimizations = [tf.lite.Optimize.DEFAULT] 
tflite_quant_model = converter.convert() 
with open('detect.tflite', "wb") as tflite_file:
     tflite_file.write(tflite_model_quant)
