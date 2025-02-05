import onnx
import onnx_tf
import tensorflow as tf

onnx_model = onnx.load("model.onnx")

tf_rep = onnx_tf.backend.prepare(onnx_model)

tf_rep.export_graph("saved_model_directory")
