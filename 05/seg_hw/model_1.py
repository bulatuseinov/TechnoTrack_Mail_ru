import torch
from torchvision.models import vgg13

"""

В качестве кодировщика пользуемся предобученными блоками сети VGG13.

"""
class VGG13Encoder(torch.nn.Module):
    def __init__(self, num_blocks, pretrained=True):
        super().__init__()
        self.num_blocks = num_blocks
        self.blocks = []
        feature_extractor = vgg13(pretrained=pretrained).features
        for i in range(self.num_blocks):
            self.blocks.append(
                torch.nn.Sequential(*[feature_extractor[j]
                                      for j in range(i * 5, i * 5 + 4)])) # берем из VGG13 все кроме maxpooling'а
                                                                          # чтобы собрать одинаковые блоки

    def forward(self, x):
        activations = []
        for i in range(self.num_blocks):
            x = self.blocks[i](x)
            activations.append(x)
            if i != self.num_blocks - 1:
                x = torch.functional.F.max_pool2d(x, kernel_size=2, stride=2)# делаем пропущенные раньше maxpoiling'и
        return activations


"""

 В нижней (центральной) части UNet решили сделать подобие Residual block'а как у призера соревнования Carvana Kaggle
 Последовательные конволюции с разными dilation и в итоге их сумма

"""

class res_block(torch.nn.Module):

    def __init__(self, num_filters, num_blocks, depth):
        super().__init__()
        self.num_filters = num_filters
        self.num_blocks = num_blocks
        self.depth = depth
        self.dilated_layers = []

        self.convs = []

        channels = self.num_filters * 2 ** self.num_blocks
        for i in range(depth):
            self.convs.append(torch.nn.Conv2d(in_channels= channels, kernel_size=(3,3),
            out_channels=channels, dilation=2**i, padding=(2**i, 2**i), stride=(1,1)))
            
    def forward(self, x):
        adds = []
        for i in range(self.depth):
            x = self.convs[i](x)
            adds.append(torch.FloatTensor(x.shape).copy_(x))

            sigma = torch.zeros_like(x.shape)

        for i in range(self.depth):
            sigma += adds[i]

        return sigma

"""

Блок декодера также сосотоит из конволюций и увеличения разрешения картинки с помощью интерполяции со scale = 2

"""
class DecoderBlock(torch.nn.Module):
    def __init__(self, out_channels):
        super().__init__()

        self.upconv = torch.nn.Conv2d(
            in_channels=out_channels * 2, out_channels=out_channels,
            kernel_size=3, padding=1, dilation=1)
        self.conv1 = torch.nn.Conv2d(
            in_channels=out_channels * 2, out_channels=out_channels,
            kernel_size=3, padding=1, dilation=1)
        self.conv2 = torch.nn.Conv2d(
            in_channels=out_channels, out_channels=out_channels,
            kernel_size=3, padding=1, dilation=1)

    def forward(self, down, left):
        x = torch.nn.functional.interpolate(down, scale_factor=2)
        x = self.upconv(x)
        x = self.conv1(torch.cat([left, x], 1))
        x = self.conv2(x)
        return x
    
class Decoder(torch.nn.Module):
    def __init__(self, num_filters, num_blocks):
        super().__init__()
        self.blocks = []
        for i in range(num_blocks):
            self.blocks.append(DecoderBlock(num_filters * 2**(num_blocks-i-1))) # увеличить изображение столько, сколько уменьшали

    def forward(self, activations):
        up = activations[-1]
        for i, left in enumerate(activations[-2::-1]):
            up = self.blocks[i](up, left)
        return up

    
class UNet(torch.nn.Module):
    def __init__(self, num_classes=1, num_filters=64, num_blocks=3):
        super().__init__()
        self.encoder = VGG13Encoder(num_blocks=num_blocks)
        self.res = res_block(num_filters, num_blocks, 4)
        self.decoder = Decoder(num_filters=64, num_blocks=num_blocks - 1)
        self.final = torch.nn.Conv2d(in_channels=num_filters, out_channels=num_classes, kernel_size=1)

    def forward(self, x):
        acts = self.encoder(x)
        x = self.decoder(acts)
        x = self.final(x)
        return x

