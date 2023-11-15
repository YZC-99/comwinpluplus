from model.backbone.resnet import resnet18,resnet34, resnet50, resnet101
import torch
data = torch.randn(4,3,256,256)

res18 = resnet18(pretrained=False)
res34 = resnet34(pretrained=False)
res50 = resnet50(pretrained=False)


"""
input:(4,3,256,256)
c1:(4,64,64,64)
c2:(4,128,32,32)
c3:(4,256,16,16)
c4:(4,512,8,8)
"""
out_18 = res18.base_forward(data)
"""
input:(4,3,256,256)
c1:(4,64,64,64)
c2:(4,128,32,32)
c3:(4,256,16,16)
c4:(4,512,8,8)
"""
out_34 = res34.base_forward(data)
"""
input:(4,3,256,256)
c1:(4,256,64,64)
c2:(4,512,32,32)
c3:(4,1024,16,16)
c4:(4,2048,8,8)
"""
out_50 = res50.base_forward(data)
print(out_18)