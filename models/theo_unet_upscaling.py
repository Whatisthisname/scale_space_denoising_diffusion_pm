import torch
import torch.functional as F
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data

# define our U-net model
class UNet_upscaling(nn.Module):
    # images have 1 channel (grayscale).

    def __init__(self, stages : int, ctx_sz : int = 11, output_scaling : int = 2):
        """`stages` defines the number of downsampling and upsampling stages. 0 stages means no downsampling or upsampling."""
        super(UNet_upscaling, self).__init__()
        self.stages = stages
        
        c_mult = 16

        self.context_size = ctx_sz
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()

        self.encoders.append(EncoderBlock(1, c_mult, ctx_sz, False))
        self.encoders.extend([EncoderBlock(c_mult * (2 ** (i)), c_mult * (2 ** (i+1)), ctx_sz) for i in range(stages)])
        
        
        if stages == 0:
            self.decoders.append(DecoderBlock(c_mult,  c_mult, ctx_sz, False))
        else:
            self.decoders.append(DecoderBlock(c_mult * 2 **(stages),  c_mult * 2 **(stages-1), ctx_sz, True))
            self.decoders.extend([DecoderBlock(2 * c_mult * (2 ** (i)), c_mult * (2 ** (i-1)), ctx_sz) for i in range(stages-1, 0, -1)])
            self.decoders.append(DecoderBlock(2*c_mult,  c_mult, ctx_sz, True))
        
        self.final_conv = nn.Conv2d(c_mult, 1, kernel_size=3, padding=1)

    def forward(self, x : torch.Tensor, context : torch.Tensor = None) -> torch.Tensor:
        # define the forward pass using skip connections
        
        # print("unet input:", x[0,0,0,0])

        # encoder
        intermediate_encodings = []
        for i in range(self.stages+1):
            # print("unet x", i, x[0,0,0,0])
            # print("x shape: {}".format(x.shape))
            x = self.encoders[i](x, context)
            intermediate_encodings.append(x)
            # print ("after encoder", i, x.shape)
        intermediate_encodings.pop() # we don't need to concatenate the last layer as it goes directly to the decoder

        intermediate_encodings.reverse() 
    
        # decoder
        for i in range(self.stages+1):
            # print("x shape: {}".format(x.shape))
            if i > 0:
                # concatenate the previous conv in the encoding stage to feed to the decoding (skip connection)
                x = torch.cat((x, intermediate_encodings[i-1]), dim=1)
            
            # determine upsample target size by inspecting shape of corresponding encoding layer
            if i < self.stages:
                upsample_target = intermediate_encodings[i].shape[-1]
            else:
                upsample_target = None # last layer won't be upsampled

            x = self.decoders[i](x, upsample_target)
            # print("unet x", i, x[0,0,0,0])

        x = self.final_conv(x)

        # exit()
        return x

class EncoderBlock(nn.Module):
    # takes input size and output size
    def __init__(self, in_ch : int, out_ch : int, ctx_sz : int, d_smpl : bool = True):
        super(EncoderBlock, self).__init__()
        in_ch = int(in_ch)
        out_ch = int(out_ch)
        self.downsample = d_smpl
        self.context_size = ctx_sz

        # define the layers of the encoder block
        if ctx_sz > 0:
            self.FiLM = FiLM(in_ch, ctx_sz)

        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        if d_smpl:
            self.pool = nn.MaxPool2d(2, 2)
        self.gelu = nn.GELU()
        self.batchnorm1 = nn.BatchNorm2d(out_ch)
        self.batchnorm2 = nn.BatchNorm2d(out_ch)

    def forward(self, x : torch.Tensor, context : torch.Tensor = None) -> torch.Tensor:
        if self.downsample:
            x = self.pool(x)

        if self.context_size > 0:
            x = self.FiLM(x, context)


        x = self.conv1(x)
        x = self.gelu(x)
        x = self.batchnorm1(x)

        x = self.conv2(x)
        x = self.gelu(x)
        x = self.batchnorm2(x)

        return x

class DecoderBlock(nn.Module):
    # takes input size and output size
    def __init__(self, in_ch : int, out_ch : int, ctx_sz : int, up_smpl : bool = True):
        super(DecoderBlock, self).__init__()
        in_ch = int(in_ch)
        out_ch = int(out_ch)
        
        # log input params:
        # print ("DecoderBlock: in_channels: {}, out_channels: {}".format(in_channels, out_channels), end = " ")
        # print ("upsample at end: {}".format(upsample))

        # define the layers of the decoder block
        
        self.do_upsample = up_smpl
        
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.gelu = nn.GELU()
        self.batchnorm1 = nn.BatchNorm2d(out_ch)
        self.batchnorm2 = nn.BatchNorm2d(out_ch)

    def forward(self, x : torch.Tensor, upsample_target : int) -> torch.Tensor:

        x = self.conv1(x)
        x = self.gelu(x)
        x = self.batchnorm1(x)
 
        x = self.conv2(x)
        x = self.gelu(x)
        x = self.batchnorm2(x)

        if self.do_upsample:
            x = nn.functional.interpolate(input = x, size = (upsample_target, upsample_target), mode='bilinear', align_corners=True)
        return x

class FiLM(nn.Module):
    """https://distill.pub/2018/feature-wise-transformations/"""
    def __init__(self, in_ch : int = 256, ctx_size : int = 1):
        super(FiLM, self).__init__()

        # In a convolutional network, FiLM applies a different affine transformation to each channel, consistent across spatial locations.

        # register the parameters of the affine transformation which depend on the context
        # make a network that maps from context to the affine transformation parameters
        
        self.context_embedding1 = nn.Linear(ctx_size, in_ch)
        self.gelu = nn.GELU()
        self.context_embedding2 = nn.Linear(in_ch, in_ch*2)
        
        # # set the embedding to be the identity by default
        # self.context_embedding[2].weight.data.zero_()


    def forward(self, x : torch.Tensor, ctx : torch.Tensor ) -> torch.Tensor:
        
        # get the affine transformation parameters from the context
        # print ("ctx shape: {}".format(ctx.shape))
        # print ("x shape: {}".format(x.shape))

        params = self.context_embedding1(ctx)
        params = self.gelu(params)
        params = self.context_embedding2(params)

        # apply the affine transformation to the input tensor
        # print ("params shape: {}".format(params.shape))
        gamma = params[:, :x.shape[1]]
        beta = params[:, x.shape[1]:]

        # # equivalently, using einops:
        # gamma, beta = einops.rearrange(params, 'b (g b) -> b g b', g=2)

        # apply transformation, channel-wise
        x = gamma.unsqueeze(-1).unsqueeze(-1) * x + beta.unsqueeze(-1).unsqueeze(-1)
        
        return x