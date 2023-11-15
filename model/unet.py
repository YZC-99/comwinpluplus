import torch
import torch.nn.functional as F
from torch import nn
from model.backbone.resnet import resnet18,resnet34, resnet50, resnet101
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from skimage import morphology


def mask_01_from_window_partition2D(x, window_size):
    B, H, W = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size)
    windows = x.permute(0, 1, 3, 2, 4)
    pixel_num_window = window_size * window_size
    mask_windows = torch.ones(B, H // window_size, W // window_size)
    mask_cond = ((windows.sum(dim=[3, 4]) == 0) + (windows.sum(dim=[3, 4]) == pixel_num_window))
    mask_windows[mask_cond] = 0
    mask_windows = mask_windows.view(-1).unsqueeze(-1).repeat(1, pixel_num_window)
    return mask_windows.cuda()

def mask_01_from_dilated_window_partition2D(x, window_size):
    """
    Args:
        x: (B, H, W)
        window_size (int): window size

    Returns:
        window_masks: (num_windows, Wh * Ww)
    """
    B, H, W = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size)
    windows = x.permute(0, 1, 3, 2, 4)
    pixel_num_window = window_size * window_size
    mask_windows = torch.ones(B, H // window_size, W // window_size)
    mask_cond = ((windows.sum(dim=[4]) == 0) + (windows.sum(dim=[4]) == pixel_num_window))
    mask_cond_boundary = ~mask_cond
    for b in range(B):
        mask_cond_boundary[b] = torch.from_numpy(morphology.binary_dilation(mask_cond_boundary[b].cpu(), morphology.square(3))).cuda()
    mask_cond = ~mask_cond_boundary
    mask_windows[mask_cond] = 0
    mask_windows = mask_windows.view(-1).unsqueeze(-1).repeat(1, pixel_num_window)
    return mask_windows.cuda()


def window_partition2D(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse2D(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, window_size, W // window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

class WindowAttention2D(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # H, W
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * self.window_size - 1) * (2 * self.window_size - 1), num_heads))  # (2*H-1) * (2*W-1), nH

        coords_h = torch.arange(self.window_size)
        coords_w = torch.arange(self.window_size)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, H, W
        coords_flatten = torch.flatten(coords, 1)  # 2, H*W
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, H*W, H*W
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # H*W, H*W, 2
        relative_coords[:, :, 0] += self.window_size - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size - 1
        relative_coords[:, :, 0] *= (2 * self.window_size - 1)
        relative_position_index = relative_coords.sum(-1)  # H*W, H*W

        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        B_, N, C = x.shape  # window_number, inner_window_pixel_number, channel
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)  # 3, B_, num_heads, N, c//num_head
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale  # B_, num_heads, N, c//num_head; k.transpose(-2, -1): B_, num_heads, c//num_head, N
        attn = (q @ k.transpose(-2, -1))  # B_, num_heads, N, N

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size * self.window_size, self.window_size * self.window_size, -1)  # H*W, H*W, nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, H*W, H*W
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            attn = attn + mask.unsqueeze(1)  # B_,self.num_heads, N, N
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'

    def flops(self, N):
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        # x = (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        return flops

class _EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=False):
        super(_EncoderBlock, self).__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if dropout:
            layers.append(nn.Dropout())
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        self.encode = nn.Sequential(*layers)

    def forward(self, x):
        return self.encode(x)


class _DecoderBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super(_DecoderBlock, self).__init__()
        self.decode = nn.Sequential(
            nn.Conv2d(in_channels, middle_channels, kernel_size=3),
            nn.BatchNorm2d(middle_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(middle_channels, middle_channels, kernel_size=3),
            nn.BatchNorm2d(middle_channels),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(middle_channels, out_channels, kernel_size=2, stride=2),
        )

    def forward(self, x):
        return self.decode(x)


class UNet(nn.Module):
    def __init__(self, num_classes):
        super(UNet, self).__init__()
        self.enc1 = _EncoderBlock(3, 64)
        self.enc2 = _EncoderBlock(64, 128)
        self.enc3 = _EncoderBlock(128, 256)
        self.enc4 = _EncoderBlock(256, 512, dropout=True)
        self.center = _DecoderBlock(512, 1024, 512)
        self.dec4 = _DecoderBlock(1024, 512, 256)
        self.dec3 = _DecoderBlock(512, 256, 128)
        self.dec2 = _DecoderBlock(256, 128, 64)
        self.dec1 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)
        center = self.center(enc4)
        dec4 = self.dec4(torch.cat([center, F.interpolate(enc4, center.size()[2:], mode='bilinear')], 1))
        dec3 = self.dec3(torch.cat([dec4, F.interpolate(enc3, dec4.size()[2:], mode='bilinear')], 1))
        dec2 = self.dec2(torch.cat([dec3, F.interpolate(enc2, dec3.size()[2:], mode='bilinear')], 1))
        dec1 = self.dec1(torch.cat([dec2, F.interpolate(enc1, dec2.size()[2:], mode='bilinear')], 1))
        final = self.classifier(dec1)
        out =  F.interpolate(final, x.size()[2:], mode='bilinear')


        return {'out':out}

class ResUNet(nn.Module):
    def __init__(self, num_classes,bb_pretrained=True,inplace_seven=False):
        super(ResUNet, self).__init__()
        self.backbone = resnet50(pretrained=bb_pretrained, inplace_seven=inplace_seven)
        self.center = _DecoderBlock(2048, 2048, 2048)
        self.dec5 = _DecoderBlock(4096, 2048, 1024)
        self.dec4 = _DecoderBlock(2048, 1024, 512)
        self.dec3 = _DecoderBlock(1024, 512, 256)
        self.dec2 = _DecoderBlock(512, 256, 128)
        self.dec1 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Conv2d(64, num_classes, kernel_size=1)
        self.classifier2 = nn.Conv2d(64, num_classes, kernel_size=1)

        self.tanh =  nn.Tanh()

    def encoder(self,input):
        # c1(2,256,64,64)
        # c2(2,512,32,32)
        # c3(2,1024,32,32)
        # c4(2,2048,32,32)
        features = self.backbone.base_forward(input)
        return [features['c1'],features['c2'],features['c3'],features['c4']]

    def decoder(self,center,features):
        dec5 = self.dec5(torch.cat([center, F.interpolate(features[3], center.size()[2:], mode='bilinear')], 1))
        dec4 = self.dec4(torch.cat([dec5, F.interpolate(features[2], dec5.size()[2:], mode='bilinear')], 1))
        dec3 = self.dec3(torch.cat([dec4, F.interpolate(features[1], dec4.size()[2:], mode='bilinear')], 1))
        dec2 = self.dec2(torch.cat([dec3, F.interpolate(features[0], dec3.size()[2:], mode='bilinear')], 1))
        dec1 = self.dec1(dec2)
        return  dec1

    def forward(self, x):
        features = self.encoder(x)
        center = self.center(features[-1])
        dec1 = self.decoder(center,features)
        # dec2 128维度
        out = self.classifier(dec1)
        out_tanh = self.tanh(out)
        out_seg = self.classifier2(dec1)

        out_tanh =  F.interpolate(out_tanh, x.size()[2:], mode='bilinear')
        out_seg =  F.interpolate(out_seg, x.size()[2:], mode='bilinear')

        return {'out_seg':out_seg,
                'out_tanh':out_tanh
                }

class ResUNet_dsba_before2(nn.Module):
    def __init__(self, num_classes,backbone = 'resnet18',bb_pretrained=True,n_filters=16,window_size=2,self_atten_head_num = 1,sparse_attn=False):
        super(ResUNet_dsba_before2, self).__init__()

        backbone_zoo = {'resnet18':resnet18,'resnet34':resnet34,'resnet50':resnet50,'resnet101':resnet101}
        self.window_size = window_size
        self.sparse_attn = sparse_attn
        self.window_eight = WindowAttention2D(n_filters * 2, window_size, num_heads=self_atten_head_num)

        self.backbone = backbone_zoo[backbone](pretrained=bb_pretrained)

        decoder_inchannels = 2048
        if backbone == 'resnet18' or backbone == 'resnet34':
            decoder_inchannels = 512

        self.center = _DecoderBlock(decoder_inchannels, decoder_inchannels, decoder_inchannels)
        self.dec5 = _DecoderBlock(decoder_inchannels * 2, decoder_inchannels, decoder_inchannels / 2)
        self.dec4 = _DecoderBlock(decoder_inchannels, decoder_inchannels / 2, decoder_inchannels / 4)
        self.dec3 = _DecoderBlock(decoder_inchannels / 2, decoder_inchannels / 4, decoder_inchannels / 8)

        self.block_two_out = nn.Conv2d(decoder_inchannels / 8, num_classes, kernel_size=1)

        self.dec2 = _DecoderBlock(decoder_inchannels / 4, decoder_inchannels / 8, decoder_inchannels / 16)
        self.dec1 = nn.Sequential(
            nn.Conv2d(decoder_inchannels / 16, decoder_inchannels / 32, kernel_size=3),
            nn.BatchNorm2d(decoder_inchannels / 32),
            nn.ReLU(inplace=True),
            nn.Conv2d(decoder_inchannels / 32, decoder_inchannels / 32, kernel_size=3),
            nn.BatchNorm2d(decoder_inchannels / 32),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(decoder_inchannels / 32, decoder_inchannels / 32, kernel_size=2, stride=2),
        )

        self.classifier = nn.Conv2d(decoder_inchannels / 32, num_classes, kernel_size=1)
        self.classifier2 = nn.Conv2d(decoder_inchannels / 32, num_classes, kernel_size=1)

        ######
        # self.center = _DecoderBlock(2048, 2048, 2048)
        # self.dec5 = _DecoderBlock(4096, 2048, 1024)
        # self.dec4 = _DecoderBlock(2048, 1024, 512)
        # self.dec3 = _DecoderBlock(1024, 512, 256)
        #
        # self.block_two_out = nn.Conv2d(256,num_classes, kernel_size=1)
        #
        # self.dec2 = _DecoderBlock(512, 256, 128)
        # self.dec1 = nn.Sequential(
        #     nn.Conv2d(128, 64, kernel_size=3),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(64, 64, kernel_size=3),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(inplace=True),
        #     nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2),
        # )
        #
        #
        # self.classifier = nn.Conv2d(64, num_classes, kernel_size=1)
        # self.classifier2 = nn.Conv2d(64, num_classes, kernel_size=1)

        self.tanh =  nn.Tanh()

    def attention_at2(self, x8, pseudo_labels):
        # todo: partition input into windows
        B, C, H, W = x8.size()
        if H % self.window_size != 0 or W % self.window_size != 0:
            padding = True
            H_ = (H // self.window_size + 1) * self.window_size
            W_ = (W // self.window_size + 1) * self.window_size
            # x8 = x8.view(-1, H, W)
            padding_op = nn.ReplicationPad2d((0, W_ - W, 0, H_ - H))
            x8 = padding_op(x8)
            pseudo_labels = padding_op(pseudo_labels.float().unsqueeze(0)).squeeze(0).long()
        else:
            padding = False

        x8_windows = window_partition2D(x8.permute(0, 2, 3, 1), self.window_size)  # nW*B, window_size, window_size, C
        x8_ = x8_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # todo: partition pseudo labels into windows
        if not self.sparse_attn:
            near_boundary_mask_eight = None
            x8_atten = self.window_eight(x8_, mask=near_boundary_mask_eight)
            # merge windows
            attn_windows = x8_atten.view(-1, self.window_size, self.window_size, C)  # (B_, N, C) -> (B_, ws, ws, C)
        else:
            if self.dilated_windows:
                x8_atten = self.window_eight(x8_)  # B_, N, C
                near_boundary_mask_eight = mask_01_from_dilated_window_partition2D(pseudo_labels,
                                                                                   self.window_size)  # B_, N
                attn_windows = (x8_atten * near_boundary_mask_eight.unsqueeze(-1)).view(-1, self.window_size,
                                                                                        self.window_size,
                                                                                        C)  # (B_, N, C) -> (B_, ws, ws, C)
            else:
                x8_atten = self.window_eight(x8_)  # B_, N, C
                near_boundary_mask_eight = mask_01_from_window_partition2D(pseudo_labels, self.window_size)  # B_, N
                attn_windows = (x8_atten * near_boundary_mask_eight.unsqueeze(-1)).view(-1, self.window_size,
                                                                                        self.window_size,
                                                                                        C)  # (B_, N, C) -> (B_, ws, ws, C)

        # x8_after = torch.cat([x8, x8_atten], dim = 1) # todo: concate or add?
        if padding:
            x8_atten_rev = window_reverse2D(attn_windows, self.window_size, H_, W_)  # B*nW, ws, ws, C -> B, H, W, C
            x8_after = x8 + x8_atten_rev.permute(0, 3, 1, 2)
            x8_after = x8_after[:, :, :H, :W]
        else:
            x8_atten_rev = window_reverse2D(attn_windows, self.window_size, H, W)  # B*nW, ws, ws, C -> B, H, W, C
            x8_after = x8 + x8_atten_rev.permute(0, 3, 1, 2)
        return x8_after

    def encoder(self,input):
        # c1(2,256,64,64)
        # c2(2,512,32,32)
        # c3(2,1024,32,32)
        # c4(2,2048,32,32)
        features = self.backbone.base_forward(input)
        return [features['c1'],features['c2'],features['c3'],features['c4']]

    # def decoder(self,center,features):
    #     dec5 = self.dec5(torch.cat([center, F.interpolate(features[3], center.size()[2:], mode='bilinear')], 1))
    #     dec4 = self.dec4(torch.cat([dec5, F.interpolate(features[2], dec5.size()[2:], mode='bilinear')], 1))
    #     dec3 = self.dec3(torch.cat([dec4, F.interpolate(features[1], dec4.size()[2:], mode='bilinear')], 1))
    #     dec2 = self.dec2(torch.cat([dec3, F.interpolate(features[0], dec3.size()[2:], mode='bilinear')], 1))
    #     dec1 = self.dec1(dec2)
    #     return  dec1

    def decoder_before2(self,center,features):
        x1 = features[0]
        dec5 = self.dec5(torch.cat([center, F.interpolate(features[3], center.size()[2:], mode='bilinear')], 1))
        dec4 = self.dec4(torch.cat([dec5, F.interpolate(features[2], dec5.size()[2:], mode='bilinear')], 1))
        dec3 = self.dec3(torch.cat([dec4, F.interpolate(features[1], dec4.size()[2:], mode='bilinear')], 1))
        out_at2 = self.block_two_out(dec3)
        out_at2 = F.interpolate(out_at2,size=[i * 2 for i in x1.size()[2:]],mode='bilinear')
        return x1, dec3, out_at2

    def decoder_after2(self,x1,dec3_after):
        dec2 = self.dec2(torch.cat([dec3_after, F.interpolate(x1, dec3_after.size()[2:], mode='bilinear')], 1))
        dec1 = self.dec1(dec2)
        return dec1

    def forward(self, x,place_holder1 = None, step=1):
        if step == 1:
            features = self.encoder(x)
            center = self.center(features[-1])
            x1, dec3, out_at2 = self.decoder_before2(center,features)
            return x1, dec3, out_at2
        elif step == 2:
            pseudo_labels = place_holder1
            # dec3_after = self.attention_at2(x, pseudo_labels)  # x8 = input
            # todo: 2d的还没实现
            dec3_after = x  # x8 = input
            return dec3_after
        else:
            assert step == 3
            x1, dec3_after = x, place_holder1
            dec1 = self.decoder_after2(x1, dec3_after)
            out = self.classifier(dec1)
            out_tanh = self.tanh(out)
            out_seg = self.classifier2(dec1)

            out_tanh =  F.interpolate(out_tanh, x.size()[2:], mode='bilinear')
            out_seg =  F.interpolate(out_seg, [i * 4 for i in x.size()[2:]], mode='bilinear')

            return out_seg
            # return {'out_seg':out_seg,
            #         'out_tanh':out_tanh
            #         }

if __name__ == '__main__':
    # resunet = ResUNet(num_classes=3,bb_pretrained=False,inplace_seven=False)
    # # model = smp.Unet(encoder_name='')
    # input = torch.randn(2,3,256,256)
    # out = resunet(input)
    # out_seg_logits = out['out_seg']
    # out_tanh_logits = out['out_tanh']
    # print(out_seg_logits.shape)
    # print(out_tanh_logits.shape)

    model = ResUNet_dsba_before2(num_classes=3,bb_pretrained=False)
    print(model)