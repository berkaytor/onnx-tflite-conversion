import tensorflow as tf

saved_model_dir = "saved_model_directory"

converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)

converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,
    tf.lite.OpsSet.SELECT_TF_OPS
]

tflite_model = converter.convert()

with open("model.tflite", "wb") as f:
    f.write(tflite_model)

print("Conversion successful!")
