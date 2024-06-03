import torch
import torch.nn as nn
import warnings
import math

from .DefConv import DeformConv3d

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


class cross_attention(nn.Module):
    def __init__(self, dim, num_heads, dropout=0.):
        super(cross_attention, self).__init__()
        if dim % num_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (dim, num_heads)
            )
        self.num_heads = num_heads
        self.attention_head_size = int(dim / num_heads)

        self.query = Depth_conv(in_ch=dim, out_ch=dim)
        self.key = Depth_conv(in_ch=dim, out_ch=dim)
        self.value = Depth_conv(in_ch=dim, out_ch=dim)

        self.dropout = nn.Dropout(dropout)

    def transpose_for_scores(self, x):
        '''
        new_x_shape = x.size()[:-1] + (
            self.num_heads,
            self.attention_head_size,
        )
        print(new_x_shape)
        x = x.view(*new_x_shape)
        '''
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, ctx):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(ctx)
        mixed_value_layer = self.value(ctx)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        attention_probs = self.dropout(attention_probs)

        ctx_layer = torch.matmul(attention_probs, value_layer)
        ctx_layer = ctx_layer.permute(0, 2, 1, 3).contiguous()

        return ctx_layer


class Depth_conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Depth_conv, self).__init__()
        self.depth_conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=in_ch,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=1,
            groups=in_ch
        )
        self.point_conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=0,
            groups=1
        )

    def forward(self, input):
        out = self.depth_conv(input)
        out = self.point_conv(out)
        return out


class Dilated_Resblock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Dilated_Resblock, self).__init__()

        sequence = list()
        sequence += [
            nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1),
                      padding=1, dilation=(1, 1)),
            nn.LeakyReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), stride=(1, 1),
                      padding=2, dilation=(2, 2)),
            nn.LeakyReLU(),
            nn.Conv2d(out_channels, in_channels, kernel_size=(3, 3), stride=(1, 1),
                      padding=3, dilation=(3, 3)),
            nn.LeakyReLU(),
            nn.Conv2d(out_channels, in_channels, kernel_size=(3, 3), stride=(1, 1),
                      padding=2, dilation=(2, 2)),
            nn.LeakyReLU(),
            nn.Conv2d(out_channels, in_channels, kernel_size=(3, 3), stride=(1, 1),
                      padding=1, dilation=(1, 1))

        ]

        self.model = nn.Sequential(*sequence)

    def forward(self, x):
        out = self.model(x) + x

        return out

class DeformConv3D_Block(nn.Module):
    def __init__(self, inp_feat, out_feat, kernel_size=3, stride=1, padding=1, bias=False):
        super(DeformConv3D_Block, self).__init__()
        self.deform_conv = DeformConv3d(inp_feat, out_feat, kernel_size=kernel_size, stride=stride, padding=padding,
                                        bias=bias)
        self.bn = nn.BatchNorm3d(out_feat)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.deform_conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class HFRM(nn.Module):
    def __init__(self, in_channels, out_channels, layer):
        super(HFRM, self).__init__()

        self.conv_head = Depth_conv(in_channels, out_channels)

        self.dilated_block_LH = Dilated_Resblock(out_channels, out_channels)
        self.dilated_block_HL = Dilated_Resblock(out_channels, out_channels)
        self.dilated_block_HH = Dilated_Resblock(out_channels, out_channels)

        # deconv3d
        self.dcn3d_block_LH = DeformConv3D_Block(3, 3, kernel_size=3, stride=1, padding=1, bias=False)
        self.dcn3d_block_HL = DeformConv3D_Block(3, 3, kernel_size=3, stride=1, padding=1, bias=False)
        self.dcn3d_block_HH = DeformConv3D_Block(3, 3, kernel_size=3, stride=1, padding=1, bias=False)

        # interscale cross-attention
        self.cross_attention0 = cross_attention(out_channels, num_heads=8)
        self.cross_attention1 = cross_attention(out_channels, num_heads=8)
        self.cross_attention2 = cross_attention(out_channels, num_heads=8)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

        # middle conv
        self.conv_mid_1 = Depth_conv(out_channels, in_channels)
        self.conv_mid_2 = Depth_conv(in_channels, out_channels)
        self.conv_tail = Depth_conv(out_channels, in_channels)
        # fuse frames
        self.conv_out = Depth_conv(in_channels, 3)



    def forward(self, x, y, layer):

        b, c, h, w = x.shape 

        residual = x

        x = self.conv_head(x)

        # extract high-pass sub-bands
        x_HL, x_LH, x_HH = x[:b//3, ...], x[b//3:2*b//3, ...], x[2*b//3:, ...] 

        # Upsample y for interscale cross-attention
        y = self.upsample(y)
        y = self.conv_head(y)
        y_HL, y_LH, y_HH = y[:b//3, ...], y[b//3:2*b//3, ...], y[2*b//3:, ...]

        # interscale cross-attention
        x_y_HH = self.cross_attention0(y_HH, x_HH)
        x_y_HL = self.cross_attention1(y_HL, x_HL)
        x_y_LH = self.cross_attention2(y_LH, x_LH)

        if layer != 0:
            # print("==> 3D def Conv")
            x_y_HL = self.conv_mid_1(x_y_HL)
            x_y_LH = self.conv_mid_1(x_y_LH)
            x_y_HH = self.conv_mid_1(x_y_HH)

            x_y_HL = x_y_HL.view(b//3, c//3, 3, h, w).permute(0, 2, 1, 3, 4) # to (b,c,t,h,w)
            x_y_LH = x_y_LH.view(b//3, c//3, 3, h, w).permute(0, 2, 1, 3, 4) # to (b,c,t,h,w)
            x_y_HH = x_y_HH.view(b//3, c//3, 3, h, w).permute(0, 2, 1, 3, 4) # to (b,c,t,h,w)

            x_HL = self.dcn3d_block_HL(x_y_HL)
            x_LH = self.dcn3d_block_LH(x_y_LH)
            x_HH = self.dcn3d_block_HH(x_y_HH)

            x_HL = x_HL.view(b//3, c, h, w)
            x_LH = x_LH.view(b//3, c, h, w)
            x_HH = x_HH.view(b//3, c, h, w)

            x_HL = self.conv_mid_2(x_HL)
            x_LH = self.conv_mid_2(x_LH)
            x_HH = self.conv_mid_2(x_HH)
        else:
            # print("==> dilation blocks")
            x_HL = self.dilated_block_HL(x_y_HL)
            x_LH = self.dilated_block_LH(x_y_LH)
            x_HH = self.dilated_block_HH(x_y_HH)

        out = self.conv_tail(torch.cat((x_HL, x_LH, x_HH), dim=0))
        out = self.conv_out(out + residual)

        return out

