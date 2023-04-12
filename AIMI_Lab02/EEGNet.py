import torch.nn as nn
from collections import OrderedDict

# TODO implement EEGNet model
class EEGNet(nn.Module):
    def __init__(self):
        super(EEGNet, self).__init__()
        
        # Layer 1 (firstconv)
        self.firstconv = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(1, 16, kernel_size=(1, 51), stride=(1, 1), padding=(0, 25), bias=False)),
            ('batchnorm1', nn.BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))
            ]))
        
        # Layer 2 (depthwiseConv)
        self.depthwiseConv = nn.Sequential(OrderedDict([
            ('conv2', nn.Conv2d(16, 32, kernel_size=(2, 1), stride=(1, 1), groups=16, bias=False)),
            ('batchnorm2', nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)),
            ('elu2', nn.ELU(alpha=0.06)),
            ('pooling2', nn.AvgPool2d(kernel_size=(1, 4), stride=(1, 4), padding=0)),
            ('drop2', nn.Dropout(p=0.5))
            ]))
         
        # Layer 3 (separableConv)
        self.separableConv = nn.Sequential(OrderedDict([
            ('conv3', nn.Conv2d(32, 32, kernel_size=(1, 15), stride=(1, 1), padding=(0, 7), bias=False)),
            ('batchnorm3', nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)),
            ('elu3', nn.ELU(alpha=0.06)),
            ('pooling3', nn.AvgPool2d(kernel_size=(1, 8), stride=(1, 8), padding=0)),
            ('drop3', nn.Dropout(p=0.5))
            ]))

        # classify
        self.classify = nn.Sequential(OrderedDict([
            ('classify', nn.Linear(in_features=736, out_features=2, bias=True))
            ]))
        
    def forward(self, x):
        # Layer 1
        x = self.firstconv(x)
        
        # Layer 2
        x = self.depthwiseConv(x)
        
        # Layer 3
        x = self.separableConv(x)
        
        # Classify Layer
        x = x.view(x.size(0), -1)  #flatten
        x = self.classify(x)
        return x
        