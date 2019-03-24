import torch
import torch.nn as nn
import torch.nn.functional as F


class VAE(nn.Module):
    """Variational Autoencoder with spatial broadcast decoder.

    Shape:
        - Input: :math:`(N, C_{in}, H_{in}, W_{in})`
        - Output: :math:`(N, C_{in}, H_{in}, W_{in})`
    """
    def __init__(self, im_size):
        super(VAE, self).__init__()
        enc_convs = [nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3,
                           stride=2),
                 nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3,
                           stride=2),
                 nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3,
                           stride=2)]
        self.enc_convs = nn.ModuleList(enc_convs)

        mlp = [nn.Linear(in_features=3136, out_features=256),
               nn.Linear(in_features=256, out_features=32)]
        self.mlp = nn.ModuleList(mlp)

        # Coordinates for the broadcast decoder
        self.im_size = im_size
        x = torch.linspace(-1, 1, im_size)
        y = torch.clone(x)
        x_grid, y_grid = torch.meshgrid(x, y)
        # Add as constant, with extra dims for N and C
        self.register_buffer('x_grid', x_grid.view((1, 1) + x_grid.shape))
        self.register_buffer('y_grid', y_grid.view((1, 1) + y_grid.shape))

        dec_convs = [nn.Conv2d(in_channels=18, out_channels=32, kernel_size=3,
                               padding=1),
                     nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3,
                               padding=1),
                     nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3,
                               padding=1)]
        self.dec_convs = nn.ModuleList(dec_convs)
        self.last_conv = nn.Conv2d(in_channels=32, out_channels=3,
                                   kernel_size=1)

    def encoder(self, x):
        batch_size = x.size(0)
        for module in self.enc_convs:
            x = module(x)
            x = F.relu(x)

        x = x.view(batch_size, -1)
        for module in self.mlp:
            x = module(x)
            x = F.relu(x)

        return torch.chunk(x, 2, dim=1)

    def sample(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decoder(self, z):
        batch_size = z.size(0)
        # View z as 4D tensor to be tiled across new H and W dimensions
        # Shape: NxDx1x1
        z = z.view(z.shape + (1, 1))

        # Tile across to match image size
        # Shape: NxDx64x64
        z = z.expand(-1, -1, self.im_size, self.im_size)

        # Expand grids to batches and concatenate on the channel dimension
        # Shape: Nx(D+2)x64x64
        x = torch.cat((self.x_grid.expand(batch_size, -1, -1, -1),
                       self.y_grid.expand(batch_size, -1, -1, -1), z), dim=1)

        for module in self.dec_convs:
            x = module(x)
            x = F.relu(x)
        x = self.last_conv(x)

        return x

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.sample(mu, logvar)
        x_rec = self.decoder(z)

        kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=-1).mean()
        mse_loss = F.mse_loss(x_rec, x)/0.02

        return mse_loss, kl, x_rec
