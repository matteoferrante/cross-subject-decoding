import torch
from torch import  nn


class ConvEncoder(nn.Module):
    def __init__(self,conv_channels=[32,64,128],bn=False,latent_dim=512,final_dim=4):
        super().__init__()
        self.latent_dim=latent_dim
        conv_layers=[]
        for c in conv_channels:
            conv_layers.append(nn.LazyConv2d(out_channels=c,kernel_size=4,stride=2,padding=1))
            if bn:
                conv_layers.append(nn.BatchNorm2d(c))
            conv_layers.append(nn.GELU())
        conv_layers.append(nn.AdaptiveAvgPool2d(final_dim))
        conv_layers.append(nn.Flatten())
        
        conv_layers.append(nn.LazyLinear(latent_dim))
        self.model=nn.Sequential(*conv_layers)
        
    
    def forward(self,x):
        return self.model(x)
            
        
class ConvDecoder(nn.Module):
    def __init__(self,conv_channels=[64,32,16],out_channels=1,bn=False,latent_dim=512,target_side=96):
        super().__init__()
        conv_layers=[]
        
        self.downsampled_side=target_side//2**len(conv_channels)
        self.predecoder=nn.LazyLinear(conv_channels[0]*self.downsampled_side**2)
        
        
        for c in conv_channels:
            conv_layers.append(nn.LazyConvTranspose2d(out_channels=c,kernel_size=4,stride=2,padding=1))
            if bn:
                conv_layers.append(nn.BatchNorm2d(c))
            conv_layers.append(nn.GELU())
        
        conv_layers.append(nn.LazyConvTranspose2d(out_channels=out_channels,kernel_size=3,stride=1,padding=1))
        conv_layers.append(nn.Sigmoid())
        self.model=nn.Sequential(*conv_layers)
    
    def forward(self,x):
        
        #reshape features
        bs=x.shape[0]
        x=self.predecoder(x)
        x=x.view(bs,-1,self.downsampled_side,self.downsampled_side)
        
        return self.model(x)        
    
    
class AE(nn.Module):
    def __init__(self,encoder,decoder):
        super().__init__()
        self.encoder=encoder
        self.decoder=decoder
        
        
    def forward(self,x):
        
        z=self.encoder(x)
        x2=self.decoder(z)
        
        return x2
