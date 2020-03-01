import taso
import onnx

graph = taso.load_onnx('onnx_models/resnext50.onnx')
print("graph.run_time(): {}ms".format(graph.run_time()))
print("graph.run_forward(): {}ms".format(graph.run_forward()))

graph = taso.load_onnx('onnx_models/resnext50_taso.onnx')
print("graph.run_time(): {}ms".format(graph.run_time()))
print("graph.run_forward(): {}ms".format(graph.run_forward()))
