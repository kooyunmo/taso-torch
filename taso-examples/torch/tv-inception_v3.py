import torch
import torch.nn as nn

import onnx
import taso
import onnxruntime as rt
import numpy as np

import time
from datetime import datetime

from collections import namedtuple
import warnings
import torch.nn.functional as F
from torch.jit.annotations import Optional
from torch import Tensor


InceptionOutputs = namedtuple('InceptionOutputs', ['logits', 'aux_logits'])
InceptionOutputs.__annotations__ = {'logits': torch.Tensor, 'aux_logits': Optional[torch.Tensor]}

# Script annotations failed with _GoogleNetOutputs = namedtuple ...
# _InceptionOutputs set here for backwards compat
_InceptionOutputs = InceptionOutputs


def inception_v3(pretrained=False, progress=True, **kwargs):
    r"""Inception v3 model architecture from
    `"Rethinking the Inception Architecture for Computer Vision" <http://arxiv.org/abs/1512.00567>`_.
    .. note::
        **Important**: In contrast to the other models the inception_v3 expects tensors with a size of
        N x 3 x 299 x 299, so ensure your images are sized accordingly.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
        aux_logits (bool): If True, add an auxiliary branch that can improve training.
            Default: *True*
        transform_input (bool): If True, preprocesses the input according to the method with which it
            was trained on ImageNet. Default: *False*
    """
    if pretrained:
        if 'transform_input' not in kwargs:
            kwargs['transform_input'] = True
        if 'aux_logits' in kwargs:
            original_aux_logits = kwargs['aux_logits']
            kwargs['aux_logits'] = True
        else:
            original_aux_logits = True
        model = Inception3(**kwargs)
        state_dict = load_state_dict_from_url(model_urls['inception_v3_google'],
                                              progress=progress)
        model.load_state_dict(state_dict)
        if not original_aux_logits:
            model.aux_logits = False
            del model.AuxLogits
        return model

    return Inception3(**kwargs)


class Inception3(nn.Module):

    def __init__(self, num_classes=1000, aux_logits=True, transform_input=False,
                 inception_blocks=None, init_weights=True):
        super(Inception3, self).__init__()
        if inception_blocks is None:
            inception_blocks = [
                BasicConv2d, InceptionA, InceptionB, InceptionC,
                InceptionD, InceptionE, InceptionAux
            ]
        assert len(inception_blocks) == 7
        conv_block = inception_blocks[0]
        inception_a = inception_blocks[1]
        inception_b = inception_blocks[2]
        inception_c = inception_blocks[3]
        inception_d = inception_blocks[4]
        inception_e = inception_blocks[5]
        inception_aux = inception_blocks[6]

        self.aux_logits = aux_logits
        self.transform_input = transform_input
        self.Conv2d_1a_3x3 = conv_block(3, 32, kernel_size=3, stride=2)
        self.Conv2d_2a_3x3 = conv_block(32, 32, kernel_size=3)
        self.Conv2d_2b_3x3 = conv_block(32, 64, kernel_size=3, padding=1)
        self.Conv2d_3b_1x1 = conv_block(64, 80, kernel_size=1)
        self.Conv2d_4a_3x3 = conv_block(80, 192, kernel_size=3)
        self.Mixed_5b = inception_a(192, pool_features=32)
        self.Mixed_5c = inception_a(256, pool_features=64)
        self.Mixed_5d = inception_a(288, pool_features=64)
        self.Mixed_6a = inception_b(288)
        self.Mixed_6b = inception_c(768, channels_7x7=128)
        self.Mixed_6c = inception_c(768, channels_7x7=160)
        self.Mixed_6d = inception_c(768, channels_7x7=160)
        self.Mixed_6e = inception_c(768, channels_7x7=192)
        if aux_logits:
            self.AuxLogits = inception_aux(768, num_classes)
        self.Mixed_7a = inception_d(768)
        self.Mixed_7b = inception_e(1280)
        self.Mixed_7c = inception_e(2048)
        self.fc = nn.Linear(2048, num_classes)
        if init_weights:
            for m in self.modules():
                if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                    import scipy.stats as stats
                    stddev = m.stddev if hasattr(m, 'stddev') else 0.1
                    X = stats.truncnorm(-2, 2, scale=stddev)
                    values = torch.as_tensor(X.rvs(m.weight.numel()), dtype=m.weight.dtype)
                    values = values.view(m.weight.size())
                    with torch.no_grad():
                        m.weight.copy_(values)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

    def _transform_input(self, x):
        if self.transform_input:
            x_ch0 = torch.unsqueeze(x[:, 0], 1) * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
            x_ch1 = torch.unsqueeze(x[:, 1], 1) * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
            x_ch2 = torch.unsqueeze(x[:, 2], 1) * (0.225 / 0.5) + (0.406 - 0.5) / 0.5
            x = torch.cat((x_ch0, x_ch1, x_ch2), 1)
        return x

    def _forward(self, x):
        # N x 3 x 299 x 299
        x = self.Conv2d_1a_3x3(x)
        # N x 32 x 149 x 149
        x = self.Conv2d_2a_3x3(x)
        # N x 32 x 147 x 147
        x = self.Conv2d_2b_3x3(x)
        # N x 64 x 147 x 147
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # N x 64 x 73 x 73
        x = self.Conv2d_3b_1x1(x)
        # N x 80 x 73 x 73
        x = self.Conv2d_4a_3x3(x)
        # N x 192 x 71 x 71
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # N x 192 x 35 x 35
        x = self.Mixed_5b(x)
        # N x 256 x 35 x 35
        x = self.Mixed_5c(x)
        # N x 288 x 35 x 35
        x = self.Mixed_5d(x)
        # N x 288 x 35 x 35
        x = self.Mixed_6a(x)
        # N x 768 x 17 x 17
        x = self.Mixed_6b(x)
        # N x 768 x 17 x 17
        x = self.Mixed_6c(x)
        # N x 768 x 17 x 17
        x = self.Mixed_6d(x)
        # N x 768 x 17 x 17
        x = self.Mixed_6e(x)
        # N x 768 x 17 x 17
        aux_defined = self.training and self.aux_logits
        if aux_defined:
            aux = self.AuxLogits(x)
        else:
            aux = None
        # N x 768 x 17 x 17
        x = self.Mixed_7a(x)
        # N x 1280 x 8 x 8
        x = self.Mixed_7b(x)
        # N x 2048 x 8 x 8
        x = self.Mixed_7c(x)
        # N x 2048 x 8 x 8
        # Adaptive average pooling
        # x = F.adaptive_avg_pool2d(x, (1, 1))
        # N x 2048 x 1 x 1
        x = F.dropout(x, training=self.training)
        # N x 2048 x 1 x 1
        # x = torch.flatten(x, 1)
        # N x 2048
        # x = self.fc(x)
        # N x 1000 (num_classes)
        return x, aux

    @torch.jit.unused
    def eager_outputs(self, x, aux):
        # type: (Tensor, Optional[Tensor]) -> InceptionOutputs
        if self.training and self.aux_logits:
            return InceptionOutputs(x, aux)
        else:
            return x

    def forward(self, x):
        x = self._transform_input(x)
        x, aux = self._forward(x)
        aux_defined = self.training and self.aux_logits
        if torch.jit.is_scripting():
            if not aux_defined:
                warnings.warn("Scripted Inception3 always returns Inception3 Tuple")
            return InceptionOutputs(x, aux)
        else:
            return self.eager_outputs(x, aux)

class InceptionA(nn.Module):

    def __init__(self, in_channels, pool_features, conv_block=None):
        super(InceptionA, self).__init__()
        if conv_block is None:
            conv_block = BasicConv2d
        self.branch1x1 = conv_block(in_channels, 64, kernel_size=1)

        self.branch5x5_1 = conv_block(in_channels, 48, kernel_size=1)
        self.branch5x5_2 = conv_block(48, 64, kernel_size=5, padding=2)

        self.branch3x3dbl_1 = conv_block(in_channels, 64, kernel_size=1)
        self.branch3x3dbl_2 = conv_block(64, 96, kernel_size=3, padding=1)
        self.branch3x3dbl_3 = conv_block(96, 96, kernel_size=3, padding=1)

        self.branch_pool = conv_block(in_channels, pool_features, kernel_size=1)

    def _forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch5x5, branch3x3dbl, branch_pool]
        return outputs

    def forward(self, x):
        outputs = self._forward(x)
        return torch.cat(outputs, 1)


class InceptionB(nn.Module):

    def __init__(self, in_channels, conv_block=None):
        super(InceptionB, self).__init__()
        if conv_block is None:
            conv_block = BasicConv2d
        self.branch3x3 = conv_block(in_channels, 384, kernel_size=3, stride=2)

        self.branch3x3dbl_1 = conv_block(in_channels, 64, kernel_size=1)
        self.branch3x3dbl_2 = conv_block(64, 96, kernel_size=3, padding=1)
        self.branch3x3dbl_3 = conv_block(96, 96, kernel_size=3, stride=2)

    def _forward(self, x):
        branch3x3 = self.branch3x3(x)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)

        branch_pool = F.max_pool2d(x, kernel_size=3, stride=2)

        outputs = [branch3x3, branch3x3dbl, branch_pool]
        return outputs

    def forward(self, x):
        outputs = self._forward(x)
        return torch.cat(outputs, 1)


class InceptionC(nn.Module):

    def __init__(self, in_channels, channels_7x7, conv_block=None):
        super(InceptionC, self).__init__()
        if conv_block is None:
            conv_block = BasicConv2d
        self.branch1x1 = conv_block(in_channels, 192, kernel_size=1)

        c7 = channels_7x7
        self.branch7x7_1 = conv_block(in_channels, c7, kernel_size=1)
        self.branch7x7_2 = conv_block(c7, c7, kernel_size=(1, 7), padding=(0, 3))
        self.branch7x7_3 = conv_block(c7, 192, kernel_size=(7, 1), padding=(3, 0))

        self.branch7x7dbl_1 = conv_block(in_channels, c7, kernel_size=1)
        self.branch7x7dbl_2 = conv_block(c7, c7, kernel_size=(7, 1), padding=(3, 0))
        self.branch7x7dbl_3 = conv_block(c7, c7, kernel_size=(1, 7), padding=(0, 3))
        self.branch7x7dbl_4 = conv_block(c7, c7, kernel_size=(7, 1), padding=(3, 0))
        self.branch7x7dbl_5 = conv_block(c7, 192, kernel_size=(1, 7), padding=(0, 3))

        self.branch_pool = conv_block(in_channels, 192, kernel_size=1)

    def _forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch7x7 = self.branch7x7_1(x)
        branch7x7 = self.branch7x7_2(branch7x7)
        branch7x7 = self.branch7x7_3(branch7x7)

        branch7x7dbl = self.branch7x7dbl_1(x)
        branch7x7dbl = self.branch7x7dbl_2(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_3(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_4(branch7x7dbl)
        branch7x7dbl = self.branch7x7dbl_5(branch7x7dbl)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch7x7, branch7x7dbl, branch_pool]
        return outputs

    def forward(self, x):
        outputs = self._forward(x)
        return torch.cat(outputs, 1)


class InceptionD(nn.Module):

    def __init__(self, in_channels, conv_block=None):
        super(InceptionD, self).__init__()
        if conv_block is None:
            conv_block = BasicConv2d
        self.branch3x3_1 = conv_block(in_channels, 192, kernel_size=1)
        self.branch3x3_2 = conv_block(192, 320, kernel_size=3, stride=2)

        self.branch7x7x3_1 = conv_block(in_channels, 192, kernel_size=1)
        self.branch7x7x3_2 = conv_block(192, 192, kernel_size=(1, 7), padding=(0, 3))
        self.branch7x7x3_3 = conv_block(192, 192, kernel_size=(7, 1), padding=(3, 0))
        self.branch7x7x3_4 = conv_block(192, 192, kernel_size=3, stride=2)

    def _forward(self, x):
        branch3x3 = self.branch3x3_1(x)
        branch3x3 = self.branch3x3_2(branch3x3)

        branch7x7x3 = self.branch7x7x3_1(x)
        branch7x7x3 = self.branch7x7x3_2(branch7x7x3)
        branch7x7x3 = self.branch7x7x3_3(branch7x7x3)
        branch7x7x3 = self.branch7x7x3_4(branch7x7x3)

        branch_pool = F.max_pool2d(x, kernel_size=3, stride=2)
        outputs = [branch3x3, branch7x7x3, branch_pool]
        return outputs

    def forward(self, x):
        outputs = self._forward(x)
        return torch.cat(outputs, 1)


class InceptionE(nn.Module):

    def __init__(self, in_channels, conv_block=None):
        super(InceptionE, self).__init__()
        if conv_block is None:
            conv_block = BasicConv2d
        self.branch1x1 = conv_block(in_channels, 320, kernel_size=1)

        self.branch3x3_1 = conv_block(in_channels, 384, kernel_size=1)
        self.branch3x3_2a = conv_block(384, 384, kernel_size=(1, 3), padding=(0, 1))
        self.branch3x3_2b = conv_block(384, 384, kernel_size=(3, 1), padding=(1, 0))

        self.branch3x3dbl_1 = conv_block(in_channels, 448, kernel_size=1)
        self.branch3x3dbl_2 = conv_block(448, 384, kernel_size=3, padding=1)
        self.branch3x3dbl_3a = conv_block(384, 384, kernel_size=(1, 3), padding=(0, 1))
        self.branch3x3dbl_3b = conv_block(384, 384, kernel_size=(3, 1), padding=(1, 0))

        self.branch_pool = conv_block(in_channels, 192, kernel_size=1)

    def _forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch3x3 = self.branch3x3_1(x)
        branch3x3 = [
            self.branch3x3_2a(branch3x3),
            self.branch3x3_2b(branch3x3),
        ]
        branch3x3 = torch.cat(branch3x3, 1)

        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = [
            self.branch3x3dbl_3a(branch3x3dbl),
            self.branch3x3dbl_3b(branch3x3dbl),
        ]
        branch3x3dbl = torch.cat(branch3x3dbl, 1)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch3x3, branch3x3dbl, branch_pool]
        return outputs

    def forward(self, x):
        outputs = self._forward(x)
        return torch.cat(outputs, 1)


class InceptionAux(nn.Module):

    def __init__(self, in_channels, num_classes, conv_block=None):
        super(InceptionAux, self).__init__()
        if conv_block is None:
            conv_block = BasicConv2d
        self.conv0 = conv_block(in_channels, 128, kernel_size=1)
        self.conv1 = conv_block(128, 768, kernel_size=5)
        self.conv1.stddev = 0.01
        self.fc = nn.Linear(768, num_classes)
        self.fc.stddev = 0.001

    def forward(self, x):
        # N x 768 x 17 x 17
        x = F.avg_pool2d(x, kernel_size=5, stride=3)
        # N x 768 x 5 x 5
        x = self.conv0(x)
        # N x 128 x 5 x 5
        x = self.conv1(x)
        # N x 768 x 1 x 1
        # Adaptive average pooling
        x = F.adaptive_avg_pool2d(x, (1, 1))
        # N x 768 x 1 x 1
        x = torch.flatten(x, 1)
        # N x 768
        x = self.fc(x)
        # N x 1000
        return x


class BasicConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)


if __name__ == '__main__':
    model = inception_v3()
    f = open("./log-tv-inception_v3.txt", 'w')
    f.write("#######################################################################\n");
    f.write("#                Inception V3 Performance Evaluation                  #\n");
    f.write("# =================================================================== #\n");
    f.write("#    time: {}                                 #\n".format(datetime.now()));
    f.write("#######################################################################\n\n");
    example_input = torch.randn(8, 3, 299, 299)

    ex_out = model(example_input)
    print("ouput shape should be: {}".format(ex_out.size()))

    # Prepare input
    torch_test_input, test_input = list(), list()
    for i in range(500):
        np_data = np.random.randn(8, 3, 299, 299).astype('f')
        test_input.append(np_data)
        torch_test_input.append(torch.from_numpy(np_data).float())


    # Export model from PyTorch (traced)
    print("torch.onnx.export()")
    torch.onnx.export(model, example_input, "./onnx_models/inception_v3.onnx", verbose=True)
    print("##### INFERENCE with onnxruntime (before TASO: traced PyTorch) #####")
    torch_sess = rt.InferenceSession("./onnx_models/inception_v3.onnx")
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
    old_graph = taso.load_onnx("./onnx_models/inception_v3.onnx")
    #print("[before opt] taso runtime performance: {}ms".format(old_graph.run_time()))
    #taso_tensor_input = old_graph.new_input_with_value(dims=(1, 3, 299, 299))
    #numpy_input = np.random.randn(1, 3, 299, 299).astype('f')
    old_graph.build_graph()
    # warm up
    for _, data in enumerate(test_input):
        old_graph.taso_forward(data)
    # real run
    time_sum = 0
    for _, data in enumerate(test_input):
        start = time.time()
        old_graph.taso_forward(data)
        time_sum += (time.time() - start)
    print("cuDNN runtime inference time before taso: {}sec".format(time_sum / len(test_input)))
    f.write("cuDNN runtime inference time before taso: {}sec\n\n".format(time_sum / len(test_input)))


    print("taso.optimize()")
    new_graph = taso.optimize(old_graph, alpha=1.05, budget=100)
    #print("[after opt] taso runtime performance: {}ms".format(new_graph.run_time()))
    #taso_tensor_input = new_graph.new_input_with_value(dims=(1, 3, 299, 299))
    new_graph.build_graph()
    # warm up
    for _, data in enumerate(test_input):
        new_graph.taso_forward(data)
    # real run
    time_sum = 0
    for _, data in enumerate(test_input):
        start = time.time()
        new_graph.taso_forward(data)
        time_sum += (time.time() - start)
    print("cuDNN runtime inference time after taso optimization: {}sec".format(time_sum / len(test_input)))
    f.write("cuDNN runtime inference time after taso optimization: {}sec\n\n".format(time_sum / len(test_input)))

    print("taso.export_onnx()")
    new_model = taso.export_onnx(new_graph)
    onnx.save(new_model, "./onnx_models/inception_v3.onnx")
    print("onnx.load()")
    taso_model = onnx.load("./onnx_models/inception_v3.onnx")
    print("TASO model graph:\n{}".format(onnx.helper.printable_graph(taso_model.graph)))
    print("##### INFERENCE with onnxruntime (after TASO) #####")
    sess = rt.InferenceSession("./onnx_models/inception_v3.onnx")
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
