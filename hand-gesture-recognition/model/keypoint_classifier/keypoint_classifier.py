import numpy as np
import tensorflow as tf
import joblib
from utils.config import settings
from model.keypoint_classifier.transformer import MaskFeatureSelector
from function import calc_bounding_rect, calc_landmark_list, pre_process_landmark


class KeyPointClassifier(object):
    def __init__(
        self,
        model_path = settings.keypoint_classifier.model_path,
        model_type = settings.keypoint_classifier.model_type,
        num_threads=1,
    ):
        self.model_type = model_type

        if model_type == 'tflite':
            # TensorFlow Lite model initialization
            self.interpreter = tf.lite.Interpreter(
                model_path=model_path,
                num_threads=num_threads
            )
            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            self.model = None
        
        elif model_type == 'sklearn':
            # scikit-learn pipeline model initialization
            self.model = joblib.load(model_path)
            self.interpreter = None
            self.input_details = None
            self.output_details = None
        
        else:
            raise ValueError(f"Unsupported model type: {model_type}. Use 'tflite' or 'sklearn'.")
    
    def __call__(
        self,
        landmark_list,
    ):
        if self.model_type == 'tflite':
            # TensorFlow Lite inference
            input_details_tensor_index = self.input_details[0]['index']
            self.interpreter.set_tensor(
                input_details_tensor_index,
                np.array([landmark_list], dtype=np.float32))
            self.interpreter.invoke()
            output_details_tensor_index = self.output_details[0]['index']
            result = self.interpreter.get_tensor(output_details_tensor_index)
            result_index = np.argmax(np.squeeze(result))

            return result_index
        
        elif self.model_type == 'sklearn':
            # scikit-learn pipeline inference
            # Reshape input data to 2D array with one sample
            X_input = np.array(landmark_list, dtype=np.float32).reshape(1, -1)
            
            # Make prediction
            result_index = self.model.predict(X_input)
            print('result_index:', result_index[0])
            
            return result_index[0]