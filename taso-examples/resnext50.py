import taso as ts
import numpy as np
import onnx
import onnxruntime as rt

import time

def resnext_block(graph, input, strides, out_channels, groups):
    w1 = graph.new_weight(dims=(out_channels,input.dim(1),1,1))
    t = graph.conv2d(input=input, weight=w1,
                     strides=(1,1), padding="SAME",
                     activation="RELU")
    w2 = graph.new_weight(dims=(out_channels,t.dim(1)//groups,3,3))
    t = graph.conv2d(input=t, weight=w2,
                     strides=strides, padding="SAME",
                     activation="RELU")
    w3 = graph.new_weight(dims=(2*out_channels,t.dim(1),1,1))
    t = graph.conv2d(input=t, weight=w3,
                     strides=(1,1), padding="SAME")
    if (strides[0]>1) or (input.dim(1) != out_channels*2):
        w4 = graph.new_weight(dims=(out_channels*2,input.dim(1),1,1))
        input=graph.conv2d(input=input, weight=w4,
                           strides=strides, padding="SAME",
                           activation="RELU")
    return graph.relu(graph.add(input, t))


test_input = list()
for i in range(100):
    test_input.append(np.random.randn(1,3,224,224).astype('f'))

graph = ts.new_graph()
input = graph.new_input(dims=(1,3,224,224))
weight = graph.new_weight(dims=(64,3,7,7))
t = graph.conv2d(input=input, weight=weight, strides=(2,2),
                 padding="SAME", activation="RELU")
t = graph.maxpool2d(input=t, kernels=(3,3), strides=(2,2), padding="SAME")
for i in range(3):
    t = resnext_block(graph, t, (1,1), 128, 32)
strides = (2,2)
for i in range(4):
    t = resnext_block(graph, t, strides, 256, 32)
    strides = (1,1)
strides = (2,2)
for i in range(6):
    t = resnext_block(graph, t, strides, 512, 32)
    strides = (1,1)
strides = (2,2)
for i in range(3):
    t = resnext_block(graph, t, strides, 1024, 32)
    strides = (1,1)

before_model = ts.export_onnx(graph)
onnx.save(before_model, "./onnx_models/resnext50.onnx")

print("##### INFERENCE (before TASO) #####")
sess1 = rt.InferenceSession("./onnx_models/resnext50.onnx")
input_name = sess1.get_inputs()[0].name
label_name = sess1.get_outputs()[0].name

time_sum = 0
for _, data in enumerate(test_input):
    start = time.time()
    output1 = sess1.run([label_name], {input_name: data})
    #print("torch_output:\n{}".format(torch_output))
    time_sum += (time.time() - start)

print("inference time before taso: {}s".format(time_sum / len(test_input)))

print("[before] taso runtime inference time: {}ms".format(graph.run_time()))
print("taso.optimize()")
new_graph = ts.optimize(graph, alpha=1.0, budget=100)
#print("taso runtime inference time: {}ms".format(new_graph.run_time()))

print("taso.export_onnx()")
onnx_model = ts.export_onnx(new_graph)

print("onnx.checker.check_model()")
onnx.checker.check_model(onnx_model)

print("onnx.save()")
onnx.save(onnx_model, "./onnx_models/resnext50_taso.onnx")

print("onnx.load()")
taso_model = onnx.load("./onnx_models/resnext50_taso.onnx")
print("TASO modle graph: \n\n{}".format(onnx.helper.printable_graph(taso_model.graph)))


print("##### INFERENCE (after TASO) #####")
sess2 = rt.InferenceSession("./onnx_models/resnext50_taso.onnx")
input_name = sess2.get_inputs()[0].name
label_name = sess2.get_outputs()[0].name

time_sum = 0
for _, data in enumerate(test_input):
    start = time.time()
    output2 = sess2.run([label_name], {input_name: data})
    #print("taso_output:\n{}".format(taso_output))
    time_sum += (time.time() - start)

print("inference time after taso: {}s".format(time_sum / len(test_input)))
