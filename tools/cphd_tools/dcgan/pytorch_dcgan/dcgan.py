import torch.nn as nn

def weights_init(m):
    """custom weights initialization
    """
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class Generator(nn.Module):
    def __init__(self, IMAGE_CHANNEL, Z_DIM, G_HIDDEN):
        super(Generator, self).__init__()
        lay1 = G_HIDDEN * 32
        lay2 = G_HIDDEN * 16
        lay3 = G_HIDDEN * 8
        lay4 = G_HIDDEN * 4
        # lay5 = G_HIDDEN * 2
        # lay6 = G_HIDDEN * 1

        kernel = 4

        self.main = nn.Sequential(
            # 1st layer
            nn.ConvTranspose2d(Z_DIM, lay1, kernel, 1, 0, bias=False),
            nn.BatchNorm2d(lay1),
            nn.ReLU(True),
            # 2nd layer
            nn.ConvTranspose2d(lay1, lay2, kernel, 2, 1, bias=False),
            nn.BatchNorm2d(lay2),
            nn.ReLU(True),
            # 3rd layer
            nn.ConvTranspose2d(lay2, lay3, kernel, 2, 1, bias=False),
            nn.BatchNorm2d(lay3),
            nn.ReLU(True),
            # 4th layer
            nn.ConvTranspose2d(lay3, lay4, kernel, 2, 1, bias=False),
            nn.BatchNorm2d(lay4),
            nn.ReLU(True),
            # # 5th layer
            # nn.ConvTranspose2d(lay4, lay5, kernel, 2, 1, bias=False),
            # nn.BatchNorm2d(lay5),
            # nn.ReLU(True),
            # # 6th layer
            # nn.ConvTranspose2d(lay5, lay6, kernel, 2, 1, bias=False),
            # nn.BatchNorm2d(lay6),
            # nn.ReLU(True),
            # output layer
            nn.ConvTranspose2d(lay4, IMAGE_CHANNEL, kernel, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        return self.main(input)


class Discriminator(nn.Module):
    def __init__(self, IMAGE_CHANNEL, D_HIDDEN):
        super(Discriminator, self).__init__()
        lay1 = D_HIDDEN * 1
        lay2 = D_HIDDEN * 2
        lay3 = D_HIDDEN * 4
        lay4 = D_HIDDEN * 8
        # lay5 = D_HIDDEN * 16
        # lay6 = D_HIDDEN * 32

        kernel = 4

        self.main = nn.Sequential(
            # 1st layer
            nn.Conv2d(IMAGE_CHANNEL, lay1, kernel, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # 2nd layer
            nn.Conv2d(lay1, lay2, kernel, 2, 1, bias=False),
            nn.BatchNorm2d(lay2),
            nn.LeakyReLU(0.2, inplace=True),
            # 3rd layer
            nn.Conv2d(lay2, lay3, kernel, 2, 1, bias=False),
            nn.BatchNorm2d(lay3),
            nn.LeakyReLU(0.2, inplace=True),
            # 4th layer
            nn.Conv2d(lay3, lay4, kernel, 2, 1, bias=False),
            nn.BatchNorm2d(lay4),
            nn.LeakyReLU(0.2, inplace=True),
            # # 5th layer
            # nn.Conv2d(lay4, lay5, kernel, 2, 1, bias=False),
            # nn.BatchNorm2d(lay5),
            # nn.LeakyReLU(0.2, inplace=True),
            # # 6th layer
            # nn.Conv2d(lay5, lay6, kernel, 2, 1, bias=False),
            # nn.BatchNorm2d(lay6),
            # nn.LeakyReLU(0.2, inplace=True),
            # output layer
            nn.Conv2d(lay4, 1, kernel, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input).view(-1, 1).squeeze(1)

