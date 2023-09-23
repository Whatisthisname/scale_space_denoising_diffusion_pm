import torch
import torch.functional as F
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data

# define our U-net model
class UNet(nn.Module):
    # images have 1 channel (grayscale).
    # `n_downs` 

    def __init__(self, image_size : int, stages : int, context_size : int = 1):
        """`stages` defines the number of downsampling and upsampling stages. 0 stages means no downsampling or upsampling."""
        super(UNet, self).__init__()
        self.image_size = image_size
        self.stages = stages
        
        channel_inflation = 16

        self.context_size = context_size
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()

        self.encoders.append(EncoderBlock(1, channel_inflation, downsample=False))
        for i in range(stages):
                self.encoders.append(EncoderBlock(channel_inflation * (2 ** (i)), channel_inflation * (2 ** (i+1))))
        
        if stages == 0:
            self.decoders.append(DecoderBlock(channel_inflation,  channel_inflation, upsample=False))

        else:
            self.decoders.append(DecoderBlock(channel_inflation * 2 **(stages),  channel_inflation * 2 **(stages-1), upsample=True))
            for i in range(stages-1, 0, -1):
                self.decoders.append(DecoderBlock(2 * channel_inflation * (2 ** (i)), channel_inflation * (2 ** (i-1)), upsample=True))

            self.decoders.append(DecoderBlock(2*channel_inflation,  channel_inflation * 1, upsample=False))
        
        self.final_conv = nn.Conv2d(channel_inflation, 1, kernel_size=3, padding=1)

        # print ("Conv2D: in_channels: {}, out_channels: {}".format(channel_inflation, 1))


    def forward(self, x : torch.Tensor) -> torch.Tensor:
        # define the forward pass using skip connections
        
        # encoder
        intermediate_encodings = []
        for i in range(self.stages+1):
            x = self.encoders[i](x)                
            intermediate_encodings.append(x)
            # print ("after encoder", i, x.shape)
        intermediate_encodings.pop() # we don't need to concatenate the last layer as it goes directly to the decoder

        intermediate_encodings.reverse() 
        
        # decoder
        for i in range(self.stages+1):

            if i > 0:
                # concatenate the previous conv in the encoding stage to feed to the decoding (skip connection)
                x = torch.cat((x, intermediate_encodings[i-1]), dim=1)
            x = self.decoders[i](x)

            
        x = self.final_conv(x)

        return x

class EncoderBlock(nn.Module):
    # takes input size and output size
    def __init__(self, in_channels : int, out_channels : int, context_size : int, downsample : bool = True):
        super(EncoderBlock, self).__init__()
        in_channels = int(in_channels)
        out_channels = int(out_channels)
        self.downsample = downsample
        self.context_size = context_size

        # log input params:
        # print ("EncoderBlock: in_channels: {}, out_channels: {}".format(in_channels, out_channels), end = " ")
        # print ("downsample immediately: {}".format(downsample))

        # define the layers of the encoder block
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        if downsample:
            self.pool = nn.MaxPool2d(2, 2)
        self.gelu = nn.GELU()
        self.batchnorm = nn.BatchNorm2d(out_channels)

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        if self.downsample:
            x = self.pool(x)
        x = self.conv1(x)
        # print ("encoder conv1 shape: {}".format(x.shape))
        x = self.gelu(x)
        x = self.conv2(x)
        # print ("encoder conv2 shape: {}".format(x.shape))
        x = self.gelu(x)
        # x = self.batchnorm(x)
        # print ("encoder pool shape: {}".format(x.shape))
        return x

class DecoderBlock(nn.Module):
    # takes input size and output size
    def __init__(self, in_channels : int, out_channels : int, context_size : int, upsample : bool = True):
        super(DecoderBlock, self).__init__()
        in_channels = int(in_channels)
        out_channels = int(out_channels)
        self.upsample = upsample
        
        # log input params:
        # print ("DecoderBlock: in_channels: {}, out_channels: {}".format(in_channels, out_channels), end = " ")
        # print ("upsample at end: {}".format(upsample))

        # define the layers of the decoder block
        if upsample:
            self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.gelu = nn.GELU()
        # self.batchnorm = nn.BatchNorm2d(out_channels)

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        # print ("decoder befupsampled shape: {}".format(x.shape))
        # print ("decoder upsampled shape: {}".format(x.shape))
        x = self.conv1(x)
        # print ("decoder conv1 shape: {}".format(x.shape))
        x = self.gelu(x)
        x = self.conv2(x)
        # print ("decoder conv2 shape: {}".format(x.shape))
        x = self.gelu(x)
        # x = self.batchnorm(x)
        if self.upsample:
            x = self.upsample(x)
        return x