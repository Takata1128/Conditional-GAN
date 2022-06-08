from cProfile import label
import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, nz, nc, ngf, nclass, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.nclass = nclass
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz + nclass, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input, class_ids):
        one_hot = (
            torch.nn.functional.one_hot(class_ids, num_classes=self.nclass)
            .unsqueeze(-1)
            .unsqueeze(-1)
        )
        input = torch.concat([input, one_hot], dim=1)
        return self.main(input)


class Discriminator(nn.Module):
    def __init__(self, nc, ndf, nclass, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.nclass = nclass
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc + nclass, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, input, class_ids):
        one_hots = torch.nn.functional.one_hot(class_ids, num_classes=self.nclass)
        label_layers = one_hots.view(-1, self.nclass, 1, 1)
        label_layers = label_layers.expand(-1, -1, input.size(2), input.size(3))
        input = torch.concat([input, label_layers], dim=1)
        return self.main(input)
