"""model.py"""

import torch
import torch.nn as nn
#import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable


def reparametrize(mu, logvar):
    std = logvar.div(2).exp()
    eps = Variable(std.data.new(std.size()).normal_())
    return mu + std*eps


class View(nn.Module):
    def __init__(self, size):
        super(View, self).__init__()
        self.size = size

    def forward(self, tensor):
        return tensor.view(self.size)


class BetaVAE_H(nn.Module):
    """Model proposed in original beta-VAE paper(Higgins et al, ICLR, 2017)."""

    def __init__(self, z_dim=32, nc=3):
        super(BetaVAE_H, self).__init__()
        self.z_dim = z_dim
        self.nc = nc
        self.encoder = nn.Sequential(
            nn.Conv2d(nc, 32, 4, 2, 1),          # B,  32, 80, 80
            nn.LeakyReLU(True),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, 4, 2, 1),          # B,  32, 40, 40
            nn.LeakyReLU(True),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, 4, 2, 1),          # B,  64, 20, 20
            nn.LeakyReLU(True),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, 4, 2, 1),          # B,  64, 10, 10
            nn.LeakyReLU(True),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 512, 10, 1),            # B, 512,  1,  1
            nn.LeakyReLU(True),
            View((-1, 512*1*1)),                 # B, 512
            nn.Linear(512, z_dim*2),             # B, z_dim*2
        )
        self.decoder = nn.Sequential(
            nn.Linear(z_dim, 512),               # B, 512
            View((-1, 512, 1, 1)),               # B, 512,  1,  1
            nn.LeakyReLU(True),
            nn.ConvTranspose2d(512, 64, 10),     # B,  64, 10, 10
            nn.LeakyReLU(True),
            nn.BatchNorm2d(64),
            nn.ConvTranspose2d(64, 64, 4, 2, 1), # B,  64, 20, 20
            nn.LeakyReLU(True),
            nn.BatchNorm2d(64),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), # B,  32, 40, 40
            nn.LeakyReLU(True),
            nn.BatchNorm2d(32),
            nn.ConvTranspose2d(32, 32, 4, 2, 1), # B,  32, 80, 80
            nn.LeakyReLU(True),
            nn.BatchNorm2d(32),
            nn.ConvTranspose2d(32, nc, 4, 2, 1),  # B, nc, 160, 160
            nn.Softmax(dim=1)
        )

        self.weight_init()

    def weight_init(self):
        for block in self._modules:
            for m in self._modules[block]:
                kaiming_init(m)

    def forward(self, x):
        x_shape = list(x.shape)
        if (len(x_shape) == 5):
            [N, C, D, H, W] = x_shape
            new_shape = [N * D, C, H, W]
            x = torch.transpose(x, 1, 2)
            x = torch.reshape(x, new_shape)
        distributions = self._encode(x)
        mu = distributions[:, :self.z_dim]
        logvar = distributions[:, self.z_dim:]
        z = reparametrize(mu, logvar)
        x_recon = self._decode(z)

        # return x_recon, mu, logvar
        return distributions

    def _encode(self, x):
        return self.encoder(x)

    def _decode(self, z):
        return self.decoder(z)


class BetaVAE_B(BetaVAE_H):
    """Model proposed in understanding beta-VAE paper(Burgess et al, arxiv:1804.03599, 2018)."""

    def __init__(self, z_dim=32, nc=3):
        super(BetaVAE_B, self).__init__()
        self.nc = nc
        self.z_dim = z_dim

        self.encoder = nn.Sequential(
            nn.Conv2d(nc, 32, 4, 2, 1),          # B,  32, 32, 32
            nn.LeakyReLU(True),
            nn.Conv2d(32, 32, 4, 2, 1),          # B,  32, 16, 16
            nn.LeakyReLU(True),
            nn.Conv2d(32, 32, 4, 2, 1),          # B,  32,  8,  8
            nn.LeakyReLU(True),
            nn.Conv2d(32, 32, 4, 2, 1),          # B,  32,  4,  4
            nn.LeakyReLU(True),
            View((-1, 32*4*4)),                  # B, 512
            nn.Linear(32*4*4, 256),              # B, 256
            nn.LeakyReLU(True),
            nn.Linear(256, 256),                 # B, 256
            nn.LeakyReLU(True),
            nn.Linear(256, z_dim*2),             # B, z_dim*2
        )

        self.decoder = nn.Sequential(
            nn.Linear(z_dim, 256),               # B, 256
            nn.LeakyReLU(True),
            nn.Linear(256, 256),                 # B, 256
            nn.LeakyReLU(True),
            nn.Linear(256, 32*4*4),              # B, 512
            nn.LeakyReLU(True),
            View((-1, 32, 4, 4)),                # B,  32,  4,  4
            nn.ConvTranspose2d(32, 32, 4, 2, 1), # B,  32,  8,  8
            nn.LeakyReLU(True),
            nn.ConvTranspose2d(32, 32, 4, 2, 1), # B,  32, 16, 16
            nn.LeakyReLU(True),
            nn.ConvTranspose2d(32, 32, 4, 2, 1), # B,  32, 32, 32
            nn.LeakyReLU(True),
            nn.ConvTranspose2d(32, nc, 4, 2, 1), # B,  nc, 64, 64
        )
        self.weight_init()

    def weight_init(self):
        for block in self._modules:
            for m in self._modules[block]:
                kaiming_init(m)

    def forward(self, x):
        distributions = self._encode(x)
        mu = distributions[:, :self.z_dim]
        logvar = distributions[:, self.z_dim:]
        z = reparametrize(mu, logvar)
        x_recon = self._decode(z).view(x.size())

        return x_recon, mu, logvar

    def _encode(self, x):
        return self.encoder(x)

    def _decode(self, z):
        return self.decoder(z)


def kaiming_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        init.kaiming_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)


def normal_init(m, mean, std):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        m.weight.data.normal_(mean, std)
        if m.bias.data is not None:
            m.bias.data.zero_()
    elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
        m.weight.data.fill_(1)
        if m.bias.data is not None:
            m.bias.data.zero_()


if __name__ == '__main__':
    pass
