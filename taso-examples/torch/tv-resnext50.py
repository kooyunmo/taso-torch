import torch
import torch.nn as nn

import taso
import onnx
import onnxruntime as rt
import numpy as np

import time
from datetime import datetime

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class Bottleneck(nn.Module):
    expansion = 4
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        self.conv1 = conv1x1(inplanes, width)
        # self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        # self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        # self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        #out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        #out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        #out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        # self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                # norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        # x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # x = self.avgpool(x)

        '''
        # TASO currently doesn't support operators in classifiers
        x = torch.flatten(x, 1)
        x = self.fc(x)
        '''

        return x

    def forward(self, x):
        return self._forward_impl(x)


def _resnet(block, layers, pretrained, progress, **kwargs):
    model = ResNet(block, layers, **kwargs)

    return model


def resnext50_32x4d(pretrained=False, progress=True, **kwargs):
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 4
    return _resnet(Bottleneck, [3, 4, 6, 3],
                   pretrained, progress, **kwargs)



if __name__ == '__main__':
    model = resnext50_32x4d()
    f = open("./log-tv-resnext.txt", 'w')
    f.write("#######################################################################\n");
    f.write("#                   ResNeXt-50 Performance Evaluation                 #\n");
    f.write("# =================================================================== #\n");
    f.write("#    time: {}                                 #\n".format(datetime.now()));
    f.write("#######################################################################\n\n");
    example_input = torch.randn(1, 3, 224, 224)
    ex_out_size = tuple(model(example_input).size())

    # Prepare input
    torch_test_input, test_input = list(), list()
    for i in range(100):
        np_data = np.random.randn(1, 3, 224, 224).astype('f')
        test_input.append(np_data)
        torch_test_input.append(torch.from_numpy(np_data).float())


    # Export model from PyTorch (traced)
    print("torch.onnx.export()")
    torch.onnx.export(model, example_input, "./onnx_models/resnext50.onnx", verbose=True)
    print("##### INFERENCE with onnxruntime (before TASO: traced PyTorch) #####")
    torch_sess = rt.InferenceSession("./onnx_models/resnext50.onnx")
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
    old_graph = taso.load_onnx("./onnx_models/resnext50.onnx")
    #print("[before opt] taso runtime performance: {}ms".format(old_graph.run_time()))
    #taso_tensor_input = old_graph.new_input_with_value(dims=(1, 3, 224, 224))
    #numpy_input = np.random.randn(1, 3, 224, 224).astype('f')
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
    #taso_tensor_input = new_graph.new_input_with_value(dims=(1, 3, 224, 224))
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
    onnx.save(new_model, "./onnx_models/resnext50_taso.onnx")
    print("onnx.load()")
    taso_model = onnx.load("./onnx_models/resnext50_taso.onnx")
    print("TASO model graph:\n{}".format(onnx.helper.printable_graph(taso_model.graph)))
    print("##### INFERENCE with onnxruntime (after TASO) #####")
    sess = rt.InferenceSession("./onnx_models/resnext50_taso.onnx")
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
