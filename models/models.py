import torch
import timm
from torch import nn
from focal_loss import BinaryFocalLoss


class encoder(nn.Module):
    def __init__(self, model = 'convnext_base_in22k', pretrained = True):
        super(encoder, self).__init__()
        self.model = timm.create_model(model, pretrained=pretrained)
    def forward(self, img):
        return self.model.forward_features(img)

class decoder(nn.Module):
    def __init__(self):
        super(decoder, self).__init__()
        channels = [1024, 512, 256, 128, 64, 1]
        decoder_list = []
        for i in range(len(channels) - 1):
            decoder_list.append(self._conv_block(channels[i], channels[i+1]))
        self.decoder_list = nn.ModuleList(decoder_list)
    def forward(self, x):
        batch_size = x.shape[0]

        
        y = x 
        for layer in self.decoder_list:
            y = layer(y)
        
  
        return y

    
    def _conv_block(self,in_dim,out_dim,drop_rate=0):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels=in_dim,
                out_channels=out_dim,
                kernel_size=4,
                stride=2,
                padding=1,
            ), 
            nn.Dropout(drop_rate),
            nn.LeakyReLU(0.2)
        )  

class mymodel(nn.Module):
    def __init__(self):
        super(mymodel, self).__init__()
        self.decoder = decoder()
        self.encoder = encoder()

    def forward(self, x):
        y = x
        y = self.encoder(y)
        y = self.decoder(y)
        return y




if __name__ == '__main__':
    img = torch.randn(3, 3, 256, 256)
    model = mymodel()

    print(model(img).shape)

