import torch
import torch.nn as nn

import onnx
import taso
import onnxruntime as rt
import numpy as np

import time
from datetime import datetime

class AlexNet(nn.Module):
    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            # conv1
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            # conv2
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            # conv3
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            # conv4
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            # conv5
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        # x = self.avgpool(x)
        # x = torch.flatten(x, 1)
        # x = self.classifier(x)
        return x


if __name__ == '__main__':
    model = AlexNet()
    f = open("./log-tv-alexnet.txt", 'w')
    f.write("#######################################################################\n");
    f.write("#                     AlexNet Performance Evaluation                  #\n");
    f.write("# =================================================================== #\n");
    f.write("#    time: {}                                 #\n".format(datetime.now()));
    f.write("#######################################################################\n\n");
    example_input = torch.randn(1, 3, 256, 256)
    ex_out_size = tuple(model(example_input).size())
    print("ouput shape should be: {}".format(ex_out_size))

    # Prepare input
    torch_test_input, test_input = list(), list()
    for i in range(100):
        np_data = np.random.randn(1, 3, 256, 256).astype('f')
        test_input.append(np_data)
        torch_test_input.append(torch.from_numpy(np_data).float())


    # Export model from PyTorch (traced)
    print("torch.onnx.export()")
    torch.onnx.export(model, example_input, "./onnx_models/alexnet.onnx", verbose=True)
    print("##### INFERENCE with onnxruntime (before TASO: traced PyTorch) #####")
    torch_sess = rt.InferenceSession("./onnx_models/alexnet.onnx")
    input_name = torch_sess.get_inputs()[0].name
    label_name = torch_sess.get_outputs()[0].name
    # warm up
    for _, data in enumerate(test_input):
        torch_sess.run([label_name], {input_name: data})
    # real run
    time_sum = 0
    for _, data in enumerate(test_input):
        start = time.time()
        # torch_output = torch_sess.run([label_name], {input_name: data}) # d
        torch_sess.run([label_name], {input_name: data})
        # print("torch_output:\n{%.6f}".format(torch_output)) # d
        time_sum += (time.time() - start)
    print("ONNX runtime inference time before taso: {}sec".format(time_sum / len(test_input)))
    f.write("ONNX runtime inference time before taso: {}sec\n\n".format(time_sum / len(test_input)))


    print("taso.load_onnx()")
    old_graph = taso.load_onnx("./onnx_models/alexnet.onnx")
    #print("[before opt] taso runtime performance: {}ms".format(old_graph.run_time()))
    #taso_tensor_input = old_graph.new_input_with_value(dims=(1, 3, 256, 256))
    #numpy_input = np.random.randn(1, 3, 256, 256).astype('f')
    old_graph.build_graph()
    # warm up
    for _, data in enumerate(test_input):
        old_graph.taso_forward(data, ex_out_size)
    # real run
    time_sum = 0
    for _, data in enumerate(test_input):
        start = time.time()
        old_graph.taso_forward(data, ex_out_size)
        time_sum += (time.time() - start)
    print("cuDNN runtime inference time before taso: {}sec".format(time_sum / len(test_input)))
    f.write("cuDNN runtime inference time before taso: {}sec\n\n".format(time_sum / len(test_input)))


    print("taso.optimize()")
    new_graph = taso.optimize(old_graph, alpha=1.05, budget=100)
    #print("[after opt] taso runtime performance: {}ms".format(new_graph.run_time()))
    #taso_tensor_input = new_graph.new_input_with_value(dims=(1, 3, 256, 256))
    new_graph.build_graph()
    # warm up
    for _, data in enumerate(test_input):
        new_graph.taso_forward(data, ex_out_size)
    # real run
    time_sum = 0
    for _, data in enumerate(test_input):
        start = time.time()
        new_graph.taso_forward(data, ex_out_size)
        time_sum += (time.time() - start)
    print("cuDNN runtime inference time after taso optimization: {}sec".format(time_sum / len(test_input)))
    f.write("cuDNN runtime inference time after taso optimization: {}sec\n\n".format(time_sum / len(test_input)))

    print("taso.export_onnx()")
    new_model = taso.export_onnx(new_graph)
    onnx.save(new_model, "./onnx_models/alexnet_taso.onnx")
    print("onnx.load()")
    taso_model = onnx.load("./onnx_models/alexnet_taso.onnx")
    print("TASO model graph:\n{}".format(onnx.helper.printable_graph(taso_model.graph)))
    print("##### INFERENCE with onnxruntime (after TASO) #####")
    sess = rt.InferenceSession("./onnx_models/alexnet_taso.onnx")
    input_name = sess.get_inputs()[0].name
    label_name = sess.get_outputs()[0].name
    # warm up
    for _, data in enumerate(test_input):
        sess.run([label_name], {input_name: data})
    # real run
    time_sum = 0
    for _, data in enumerate(test_input):
        start = time.time()
        # taso_output = sess.run([label_name], {input_name: data}) # d
        sess.run([label_name], {input_name: data})
        # print("taso_output:\n{%.6f}".format(taso_output))
        time_sum += (time.time() - start) # d
    print("ONNX runtime inference time after taso optimization: {}sec".format(time_sum / len(test_input)))
    f.write("ONNX runtime inference time after taso optimization: {}sec\n\n".format(time_sum / len(test_input)))
    # print("same output?: {}".format(np.array_equal(torch_output, taso_output)))
    # print("diff: {}".format(np.subtract(torch_output, taso_output)))


    # measure the time of imperative torch
    # warm up
    for _, data in enumerate(torch_test_input):
        model.to(device='cuda:0')
        model(data.to(device='cuda:0'))
    # real run
    time_sum = 0
    for _, data in enumerate(torch_test_input):
        start = time.time()
        model.to(device='cuda:0')
        model(data.to(device='cuda:0'))
        time_sum += (time.time() - start)
    # print("ouput shape should be: {}".format(ex_out.size()))
    print("Imperative PyTorch inference time: {}sec".format(time_sum / len(torch_test_input)))
    f.write("Imperative PyTorch inference time: {}sec\n\n".format(time_sum / len(torch_test_input)))


    # measure the time of TorchScript
    # warm up
    model.to(device='cuda:0')
    model_script = torch.jit.script(model)
    for _, data in enumerate(torch_test_input):
        model_script(data.to(device='cuda:0'))
    # real run
    time_sum = 0
    for _, data in enumerate(torch_test_input):
        start = time.time()
        model_script(data.to(device='cuda:0'))
        time_sum += (time.time() - start)
    print("TorchScript inference time: {}sec".format(time_sum / len(torch_test_input)))
    f.write("TorchScript inference time: {}sec\n\n".format(time_sum / len(torch_test_input)))


    # measure the time of PyTorch trace
    model_trace = torch.jit.trace(model, example_input.to(device='cuda:0'))
    for _, data in enumerate(torch_test_input):
        model_trace(data.to(device='cuda:0'))
    # real run
    time_sum = 0
    for _, data in enumerate(torch_test_input):
        start = time.time()
        model_trace(data.to(device='cuda:0'))
        time_sum += (time.time() - start)
    print("PyTorch trace inference time: {}sec".format(time_sum / len(torch_test_input)))
    f.write("PyTorch trace inference time: {}sec\n\n".format(time_sum / len(torch_test_input)))

    f.close()
