import cv2
import numpy as np
import tensorflow.lite as tflite

interpreter = tflite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def preprocess_image(image_path, input_shape):
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    input_channels, input_height, input_width = input_shape[1], input_shape[2], input_shape[3]
    img = cv2.resize(img, (input_width, input_height))
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1)) 
    img = np.expand_dims(img, axis=0)
    return img


def run_inference(image_path):
    input_shape = input_details[0]['shape']
    input_data = preprocess_image(image_path, input_shape)
    input_data = input_data.astype(input_details[0]['dtype'])
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    return output_data

def save_depth_map(depth_map, output_path="depth_map.jpg"):
    depth_map = np.squeeze(depth_map)
    depth_map = np.transpose(depth_map, (1, 0))
    depth_map = cv2.rotate(depth_map, cv2.ROTATE_90_CLOCKWISE)
    depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())
    depth_map = (depth_map * 255).astype(np.uint8) 
    cv2.imwrite(output_path, depth_map)
    print(f"Depth map saved as {output_path}")

sample_image_path = "sample.png"
depth_map = run_inference(sample_image_path)

save_depth_map(depth_map, "sample.jpg")