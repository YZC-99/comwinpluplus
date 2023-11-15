from model.unet import ResUNet_dsba_before2
import torch
import torch.nn as nn

class TriUNet_before2(nn.Module):
    def __init__(self,
                 num_classes,
                 backbone = 'resnet18',
                 bb_pretrained=True):
        super().__init__()
        self.branch1 = ResUNet_dsba_before2(num_classes=num_classes,backbone=backbone,bb_pretrained=bb_pretrained)
        self.branch2 = ResUNet_dsba_before2(num_classes=num_classes,backbone=backbone,bb_pretrained=bb_pretrained)
        self.branch3 = ResUNet_dsba_before2(num_classes=num_classes,backbone=backbone,bb_pretrained=bb_pretrained)

    def forward(self, data, pseudo_labels=None, step=1, forward_step=1):
        # if not self.training:
        #     pred1 = self.branch1(data)
        #     return pred1
        if not self.training:
            """
            x1_test_1: 来自V-Net的高维特征
            x8_test_1: 是V-Net解码器的第8个块的输出
            out_at8_test_1: 是v-Net解码器第8个块输出的分类头结果
            """
            x1_test_1, x8_test_1, out_at8_test_1 = self.branch1(data, step=1)
            # max_test_1 低分辨率的伪标签
            _, max_test_1 = torch.max(out_at8_test_1, dim=1)
            # x8_after_test_1的形状与x8_test_1一致，只不过是增加了一个attention机制而已
            x8_after_test_1 = self.branch1(x8_test_1, max_test_1, step=2)
            # x1_test_1的输入主要是为了结合x8_after_test_1，因为要遵循U-Net的架构
            logits_test_1 = self.branch1(x1_test_1, x8_after_test_1, step=3)
            return logits_test_1
        if forward_step == 1:
            if step == 1:
                return self.branch1(data, step=forward_step)
            elif step == 2:
                return self.branch2(data, step=forward_step)
            elif step == 3:
                return self.branch3(data, step=forward_step)
        elif forward_step == 2:
            if step == 1:
                return self.branch1(data, pseudo_labels, step=forward_step)
            elif step == 2:
                return self.branch2(data, pseudo_labels, step=forward_step)
            elif step == 3:
                return self.branch3(data, pseudo_labels, step=forward_step)
        else:
            assert forward_step == 3
            if step == 1:
                return self.branch1(*data, step=forward_step)
            elif step == 2:
                return self.branch2(*data, step=forward_step)
            elif step == 3:
                return self.branch3(*data, step=forward_step)
if __name__ == '__main__':
    model = TriUNet_before2(num_classes=3,bb_pretrained=False)
    print(model)