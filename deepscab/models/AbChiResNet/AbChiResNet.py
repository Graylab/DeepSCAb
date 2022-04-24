import math
from os.path import isfile
import torch.nn as nn
import torch.nn.functional as F
import torch
from deepscab.resnets import ResNet1D, ResBlock1D, ResNet2D, ResBlock2D
from deepscab.layers import OuterConcatenation2D, Flatten2D


class AbChiResNet(nn.Module):
    """
    Predicts binned output distributions for distance, omega and theta dihedrals, and
    phi planar angle from a one-hot encoded sequence of heavy and light chain resides.
    """
    def __init__(self,
                 in_planes,
                 num_out_bins=36,
                 num_blocks1D=3,
                 num_blocks2D=21,
                 dilation_cycle=5,
                 dropout_proportion=0.2):
        super(AbChiResNet, self).__init__()
        if isinstance(num_blocks1D, list):
            if len(num_blocks1D) > 1:
                raise NotImplementedError('Multi-layer resnets not supported')
            num_blocks1D = num_blocks1D[0]
        if isinstance(num_blocks2D, int):
            num_blocks2D = [num_blocks2D]

        self._num_out_bins = num_out_bins
        self.resnet1D = ResNet1D(in_planes,
                                 ResBlock1D, [num_blocks1D],
                                 init_planes=32,
                                 kernel_size=17)
        self.seq_to_pairwise = OuterConcatenation2D()

        # Calculate the number of planes output from the seq2pairwise layer
        expansion1D = int(math.pow(2, self.resnet1D.num_layers - 1))
        out_planes1D = self.resnet1D.init_planes * expansion1D
        in_planes2D = 2 * out_planes1D

        self.resnet2D = ResNet2D(in_planes2D,
                                 ResBlock2D,
                                 num_blocks2D,
                                 init_planes=64,
                                 kernel_size=5,
                                 dilation_cycle=dilation_cycle)

        # Calculate the number of planes output from the ResNet2D layer
        expansion2D = int(math.pow(2, self.resnet2D.num_layers - 1))
        out_planes2D = self.resnet2D.init_planes * expansion2D

        self.out_dropout = nn.Dropout2d(p=dropout_proportion)

        self.pairwise_to_seq = Flatten2D(sum_row_col=False)

        # Output convolution to reduce/expand to the number of bins
        self.out_conv_dist_two = nn.Conv2d(
            out_planes2D + 10 * num_out_bins,
            num_out_bins,
            kernel_size=self.resnet2D.kernel_size,
            padding=self.resnet2D.kernel_size // 2)
        self.out_conv_omega_two = nn.Conv2d(
            out_planes2D + 10 * num_out_bins,
            num_out_bins,
            kernel_size=self.resnet2D.kernel_size,
            padding=self.resnet2D.kernel_size // 2)
        self.out_conv_theta_two = nn.Conv2d(
            out_planes2D + 10 * num_out_bins,
            num_out_bins,
            kernel_size=self.resnet2D.kernel_size,
            padding=self.resnet2D.kernel_size // 2)
        self.out_conv_phi_two = nn.Conv2d(
            out_planes2D + 10 * num_out_bins,
            num_out_bins,
            kernel_size=self.resnet2D.kernel_size,
            padding=self.resnet2D.kernel_size // 2)
        self.out_conv_dist = nn.Conv2d(out_planes2D,
                                       num_out_bins,
                                       kernel_size=self.resnet2D.kernel_size,
                                       padding=self.resnet2D.kernel_size // 2)
        self.out_conv_omega = nn.Conv2d(out_planes2D,
                                        num_out_bins,
                                        kernel_size=self.resnet2D.kernel_size,
                                        padding=self.resnet2D.kernel_size // 2)
        self.out_conv_theta = nn.Conv2d(out_planes2D,
                                        num_out_bins,
                                        kernel_size=self.resnet2D.kernel_size,
                                        padding=self.resnet2D.kernel_size // 2)
        self.out_conv_phi = nn.Conv2d(out_planes2D,
                                      num_out_bins,
                                      kernel_size=self.resnet2D.kernel_size,
                                      padding=self.resnet2D.kernel_size // 2)

        self.transencode_attention = nn.TransformerEncoderLayer(
            d_model=2 * out_planes2D,
            nhead=8,
            dim_feedforward=8 * out_planes2D,
            dropout=0.1)
        self.activation = nn.Softmax(dim=1)
        torch.autograd.set_detect_anomaly(True)
        self.out_conv_chi_one = nn.Conv1d(
            2 * out_planes2D,
            num_out_bins,
            kernel_size=self.resnet2D.kernel_size,
            padding=self.resnet2D.kernel_size // 2)
        self.out_conv_chi_two = nn.Conv1d(
            2 * out_planes2D + 36,
            num_out_bins,
            kernel_size=self.resnet2D.kernel_size,
            padding=self.resnet2D.kernel_size // 2)
        self.out_conv_chi_three = nn.Conv1d(
            2 * out_planes2D + 72,
            num_out_bins,
            kernel_size=self.resnet2D.kernel_size,
            padding=self.resnet2D.kernel_size // 2)
        self.out_conv_chi_four = nn.Conv1d(
            2 * out_planes2D + 108,
            num_out_bins,
            kernel_size=self.resnet2D.kernel_size,
            padding=self.resnet2D.kernel_size // 2)
        self.out_conv_chi_five = nn.Conv1d(
            2 * out_planes2D + 144,
            num_out_bins,
            kernel_size=self.resnet2D.kernel_size,
            padding=self.resnet2D.kernel_size // 2)

    def freeze_original(self, freeze=True):
        for p in self.resnet1D.parameters():
            p.requires_grad = not freeze
        for p in self.resnet2D.parameters():
            p.requires_grad = not freeze
        for p in self.out_conv_dist.parameters():
            p.requires_grad = not freeze
        for p in self.out_conv_omega.parameters():
            p.requires_grad = not freeze
        for p in self.out_conv_theta.parameters():
            p.requires_grad = not freeze
        for p in self.out_conv_phi.parameters():
            p.requires_grad = not freeze

    def forward(self, x):
        out = self.resnet1D(x)
        out = self.seq_to_pairwise(out)
        out = self.resnet2D(out)
        out = self.out_dropout(out)

        out_rotamers = self.pairwise_to_seq(out)
        out_rotamers = self.transencode_attention(out_rotamers.permute(
            2, 0, 1))
        out_rotamers = out_rotamers.permute(1, 2, 0)
        out_chi_one = self.out_conv_chi_one(out_rotamers)
        out_chi_two = self.out_conv_chi_two(
            torch.cat(
                [out_rotamers, self.activation(out_chi_one)], dim=1))
        out_chi_three = self.out_conv_chi_three(
            torch.cat([
                out_rotamers,
                self.activation(out_chi_one),
                self.activation(out_chi_two)
            ],
                      dim=1))
        out_chi_four = self.out_conv_chi_four(
            torch.cat([
                out_rotamers,
                self.activation(out_chi_one),
                self.activation(out_chi_two),
                self.activation(out_chi_three)
            ],
                      dim=1))
        out_chi_five = self.out_conv_chi_five(
            torch.cat([
                out_rotamers,
                self.activation(out_chi_one),
                self.activation(out_chi_two),
                self.activation(out_chi_three),
                self.activation(out_chi_four)
            ],
                      dim=1))

        out_rotamers = torch.cat([
            self.activation(out_chi_one),
            self.activation(out_chi_two),
            self.activation(out_chi_three),
            self.activation(out_chi_four),
            self.activation(out_chi_five)
        ],
                                 dim=1)
        out_rotamers = self.seq_to_pairwise(out_rotamers)
        out = torch.cat([out_rotamers, out], dim=1)

        out_dist = self.out_conv_dist_two(out)
        out_omega = self.out_conv_omega_two(out)
        out_theta = self.out_conv_theta_two(out)
        out_phi = self.out_conv_phi_two(out)
        out_dist = out_dist + out_dist.transpose(2, 3)
        out_omega = out_omega + out_omega.transpose(2, 3)

        return [
            out_dist, out_omega, out_theta, out_phi, out_chi_one, out_chi_two,
            out_chi_three, out_chi_four, out_chi_five
        ]

    def forward_attention(self, x):
        out = self.resnet1D(x)
        out = self.seq_to_pairwise(out)
        out = self.resnet2D(out)
        out = self.out_dropout(out)

        out_rotamers = self.pairwise_to_seq(out)
        attns = self.transencode_attention.self_attn(
            out_rotamers.permute(2, 0, 1), out_rotamers.permute(2, 0, 1),
            out_rotamers.permute(2, 0, 1))[1]

        return attns


def load_model(model_file, eval_mode=True):
    if not isfile(model_file):
        raise FileNotFoundError("No file at {}".format(model_file))
    checkpoint_dict = torch.load(model_file, map_location='cpu')
    model_state = checkpoint_dict['model_state_dict']

    dilation_cycle = 0 if not 'dilation_cycle' in checkpoint_dict else checkpoint_dict[
        'dilation_cycle']

    in_layer = list(model_state.keys())[0]
    out_layer = list(model_state.keys())[-1]
    num_out_bins = model_state[out_layer].shape[0]
    in_planes = model_state[in_layer].shape[1]

    num_blocks1D = checkpoint_dict['num_blocks1D']
    num_blocks2D = checkpoint_dict['num_blocks2D']

    model = AbChiResNet(in_planes=in_planes,
                        num_out_bins=num_out_bins,
                        num_blocks1D=num_blocks1D,
                        num_blocks2D=num_blocks2D,
                        dilation_cycle=dilation_cycle)

    model.load_state_dict(model_state, strict=False)

    if eval_mode:
        model.eval()

    return model
