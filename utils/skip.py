
import torch
import torch.nn as nn
import numpy as np


class Concat(nn.Module):
    def __init__(self, dim, *args):
        super(Concat, self).__init__()
        self.dim = dim

        for idx, module_ in enumerate(args):
            self.add_module(str(idx), module_)

    def forward(self, input_):
        inputs = []
        for module_ in self._modules.values():
            inputs.append(module_(input_))

        inputs_shapes2 = [x.shape[2] for x in inputs]
        inputs_shapes3 = [x.shape[3] for x in inputs]

        if np.all(np.array(inputs_shapes2) == min(inputs_shapes2)) and np.all(
                np.array(inputs_shapes3) == min(inputs_shapes3)):
            inputs_ = inputs
        else:
            target_shape2 = min(inputs_shapes2)
            target_shape3 = min(inputs_shapes3)

            inputs_ = []
            for inp in inputs:
                diff2 = (inp.size(2) - target_shape2) // 2
                diff3 = (inp.size(3) - target_shape3) // 2
                inputs_.append(
                    inp[:, :, diff2: diff2 + target_shape2, diff3:diff3 + target_shape3])

        return torch.cat(inputs_, dim=self.dim)

    def __len__(self):
        return len(self._modules)


def Conv(in_f, out_f, kernel_size, stride=1, bias=True, pad='zero'):
    downsampler = None
    padder = None
    to_pad = int((kernel_size - 1) / 2)
    if pad == 'reflection':
        padder = nn.ReflectionPad2d(to_pad)
        to_pad = 0

    convolver = nn.Conv2d(in_f, out_f, kernel_size,
                          stride, padding=to_pad, bias=bias)

    layers = [x for x in [padder, convolver, downsampler] if x is not None]
    return nn.Sequential(*layers)


def Skip(
        num_input_channels=2, num_output_channels=3, num_channels_down=[16, 32, 64, 128, 128],
        num_channels_up=[16, 32, 64, 128, 128],
        num_channels_skip=[4, 4, 4, 4, 4], filter_size_down=3,
        filter_size_up=3, filter_skip_size=1, need_sigmoid=True, need_bias=True,
        pad='zero', upsample_mode='nearest', need1x1_up=True):
    """
    Assembles encoder-decoder with skip connections.

    Arguments:
        act_fun: Either string 'LeakyReLU|Swish|ELU|none' or module (e.g. nn.ReLU)
        pad (string): zero|reflection (default: 'zero')
        upsample_mode (string): 'nearest|bilinear' (default: 'nearest')
        downsample_mode (string): 'stride|avg|max|lanczos2' (default: 'stride')

    """
    assert len(num_channels_down) == len(
        num_channels_up) == len(num_channels_skip)

    n_scales = len(num_channels_down)

    if not (isinstance(upsample_mode, list) or isinstance(upsample_mode, tuple)):
        upsample_mode = [upsample_mode] * n_scales

    if not (isinstance(filter_size_down, list) or isinstance(filter_size_down, tuple)):
        filter_size_down = [filter_size_down] * n_scales

    if not (isinstance(filter_size_up, list) or isinstance(filter_size_up, tuple)):
        filter_size_up = [filter_size_up] * n_scales

    last_scale = n_scales - 1

    model = nn.Sequential()
    model_tmp = model

    input_depth = num_input_channels
    for i in range(len(num_channels_down)):

        deeper = nn.Sequential()
        skip = nn.Sequential()

        if num_channels_skip[i] != 0:
            model_tmp.append(Concat(1, skip, deeper))
        else:
            model_tmp.append(deeper)

        model_tmp.append(nn.BatchNorm2d(
            num_channels_skip[i] + (num_channels_up[i + 1] if i < last_scale else num_channels_down[i])))

        if num_channels_skip[i] != 0:
            skip.append(Conv(
                input_depth, num_channels_skip[i], filter_skip_size, bias=need_bias, pad=pad))
            skip.append(nn.BatchNorm2d(num_channels_skip[i]))
            skip.append(nn.LeakyReLU(0.2, inplace=True))

        deeper.append(Conv(
            input_depth, num_channels_down[i], filter_size_down[i], 2, bias=need_bias, pad=pad))
        deeper.append(nn.BatchNorm2d(num_channels_down[i]))
        deeper.append(nn.LeakyReLU(0.2, inplace=True))

        deeper.append(Conv(
            num_channels_down[i], num_channels_down[i], filter_size_down[i], bias=need_bias, pad=pad))
        deeper.append(nn.BatchNorm2d(num_channels_down[i]))
        deeper.append(nn.LeakyReLU(0.2, inplace=True))

        deeper_main = nn.Sequential()

        if i == len(num_channels_down) - 1:
            # The deepest
            k = num_channels_down[i]
        else:
            deeper.append(deeper_main)
            k = num_channels_up[i + 1]

        deeper.append(nn.Upsample(scale_factor=2,
                      mode=upsample_mode[i], align_corners=True))

        model_tmp.append(Conv(
            num_channels_skip[i] + k, num_channels_up[i], filter_size_up[i], 1, bias=need_bias, pad=pad))
        model_tmp.append(nn.BatchNorm2d(num_channels_up[i]))
        # model_tmp.append(layer_norm(num_channels_up[i]))
        model_tmp.append(nn.LeakyReLU(0.2, inplace=True))

        if need1x1_up:
            model_tmp.append(
                Conv(num_channels_up[i], num_channels_up[i], 1, bias=need_bias, pad=pad))
            model_tmp.append(nn.BatchNorm2d(num_channels_up[i]))
            # model_tmp.append(layer_norm(num_channels_up[i]))
            model_tmp.append(nn.LeakyReLU(0.2, inplace=True))

        input_depth = num_channels_down[i]
        model_tmp = deeper_main

    model.append(
        Conv(num_channels_up[0], num_output_channels, 1, bias=need_bias, pad=pad))
    if need_sigmoid:
        model.append(nn.Sigmoid())
    return model
