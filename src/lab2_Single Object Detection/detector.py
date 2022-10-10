from turtle import forward
import torch.nn as nn
import resnet


# TODO Design the detector.
# tips: Use pretrained `resnet` as backbone.
class NeuralNetwork(nn.Module):
    def __init__(self, lengths, num_classes):
        super().__init__()
        lst=[]
        for i in range(len(lengths)-1):
            lst.append(nn.Linear(lengths[i],lengths[i+1]))
            lst.append(nn.ReLU())
        lst1=lst
        lst2=lst
        lst1.append(nn.Linear(lengths[-1],num_classes))
        self.cls_pred=nn.Sequential(*tuple(lst1))
        lst.pop()
        lst2.append(nn.Linear(lengths[-1],4))
        self.bbox_pred=nn.Sequential(*tuple(lst2))
    def forward(self,x):
        logits=self.cls_pred(x)
        bbox=self.bbox_pred(x)
        return logits,bbox


class Detector(nn.Module):
    def __init__(self, backbone, lengths, num_classes):
        super().__init__()
        self.backbone=getattr(resnet,backbone)(pretrained=True)
        self.mynet=NeuralNetwork(lengths, num_classes)
        self.flat=nn.Flatten()
    def forward(self,x):
        x=self.backbone(x)
        x=self.flat(x)
        logits,bbox=self.mynet(x)
        return logits,bbox
...

# End of todo
