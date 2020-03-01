import torch
import torch.nn as nn

import taso
import onnx
import onnxruntime as rt
import numpy as np

import time
from datetime import datetime


class VGG(nn.Module):

    def __init__(self, features, num_classes=1000, init_weights=True):
        super(VGG, self).__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        #x = self.avgpool(x)
        #x = torch.flatten(x, 1)
        #x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfgs = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def _vgg(arch, cfg, batch_norm, pretrained, progress, **kwargs):
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(make_layers(cfgs[cfg], batch_norm=batch_norm), **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model


def vgg19(pretrained=False, progress=True, **kwargs):
    r"""VGG 19-layer model (configuration "E")
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg19', 'E', False, pretrained, progress, **kwargs)


if __name__ == '__main__':
    model = vgg19()
    f = open("./log-tv-vgg19.txt", 'w')
    f.write("#######################################################################\n");
    f.write("#                     VGG-19 Performance Evaluation                   #\n");
    f.write("# =================================================================== #\n");
    f.write("#    time: {}                                 #\n".format(datetime.now()));
    f.write("#######################################################################\n\n");
    example_input = torch.randn(1, 3, 224, 224)
    ex_out_size = tuple(model(example_input).size())
    print("ouput shape should be: {}".format(ex_out_size))

    # Prepare input
    torch_test_input, test_input = list(), list()
    for i in range(100):
        np_data = np.random.randn(1, 3, 224, 224).astype('f')
        test_input.append(np_data)
        torch_test_input.append(torch.from_numpy(np_data).float())


    # Export model from PyTorch (traced)
    print("torch.onnx.export()")
    torch.onnx.export(model, example_input, "./onnx_models/vgg19.onnx", verbose=True)
    print("##### INFERENCE with onnxruntime (before TASO: traced PyTorch) #####")
    torch_sess = rt.InferenceSession("./onnx_models/vgg19.onnx")
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
    old_graph = taso.load_onnx("./onnx_models/vgg19.onnx")
    #print("[before opt] taso runtime performance: {}ms".format(old_graph.run_time()))
    #taso_tensor_input = old_graph.new_input_with_value(dims=(1, 3, 224, 224))
    #numpy_input = np.random.randn(1, 3, 224, 224).astype('f')
    old_graph.build_graph()
    # warm up
    for _, data in enumerate(test_input):
        res1 = old_graph.taso_forward(data, ex_out_size)
    # real run
    time_sum = 0
    for _, data in enumerate(test_input):
        start = time.time()
        res1 = old_graph.taso_forward(data, ex_out_size)
        time_sum += (time.time() - start)
    print("cuDNN runtime inference time before taso: {}sec".format(time_sum / len(test_input)))
    f.write("cuDNN runtime inference time before taso: {}sec\n\n".format(time_sum / len(test_input)))


    print("taso.optimize()")
    new_graph = taso.optimize(old_graph, alpha=1.05, budget=100)
    #print("[after opt] taso runtime performance: {}ms".format(new_graph.run_time()))
    #taso_tensor_input = new_graph.new_input_with_value(dims=(1, 3, 224, 224))
    new_graph.build_graph()
    # warm up
    for _, data in enumerate(test_input):
        res2 = new_graph.taso_forward(data, ex_out_size)
    # real run
    time_sum = 0
    for _, data in enumerate(test_input):
        start = time.time()
        res2 = new_graph.taso_forward(data, ex_out_size)
        time_sum += (time.time() - start)
    print("cuDNN runtime inference time after taso optimization: {}sec".format(time_sum / len(test_input)))
    f.write("cuDNN runtime inference time after taso optimization: {}sec\n\n".format(time_sum / len(test_input)))

    print("diff: {}".format(np.subtract(res1, res2)))

    print("taso.export_onnx()")
    new_model = taso.export_onnx(new_graph)
    onnx.save(new_model, "./onnx_models/vgg19_taso.onnx")
    print("onnx.load()")
    taso_model = onnx.load("./onnx_models/vgg19_taso.onnx")
    print("TASO model graph:\n{}".format(onnx.helper.printable_graph(taso_model.graph)))
    print("##### INFERENCE with onnxruntime (after TASO) #####")
    sess = rt.InferenceSession("./onnx_models/vgg19_taso.onnx")
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
    #print("ouput shape should be: {}".format(ex_out.size()))
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
