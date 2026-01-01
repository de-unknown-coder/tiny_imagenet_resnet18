import torch
import torch.nn as nn 
import torch.nn.functional as F 
from ResNet18_Blocks import basicBlock

class ResNet18(nn.Module):
    def __init__(self,num_classes=200):
        super().__init__()
        
        self.in_ch=64
        
        self.stem = nn.Sequential(
            nn.Conv2d(3,64,7,2,3,bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(3,2,1)
        )
        
        channels=[64,128,256,512]
        blocks = [2,2,2,2]
        stride=[1,2,2,2]
        
        self.stages= nn.ModuleList()
        
        for out_ch, num_blocks, stride in zip(channels,blocks,stride):
            stage=[]
            
            stage.append(basicBlock(self.in_ch,out_ch, stride))
            self.in_ch=out_ch
            
            for _ in range(1,num_blocks):
                stage.append(basicBlock(self.in_ch,out_ch,1))
            
            self.stages.append(nn.Sequential(*stage))
        
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc =nn.Linear(512,num_classes)
    
    def forward(self, x):
        x= self.stem(x)
        for stage in self.stages:
            x=stage(x)
        x=self.gap(x)
        x=torch.flatten(x,1)
        return self.fc(x)
    

        
    
     