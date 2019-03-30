import torch
from models import VAE
from utils import plot_examples

model = VAE(im_size=64)
model.load_state_dict(torch.load('vae_sbd.pt', map_location='cpu'))
data = torch.load('data/train.pt')[:8]
plot_examples(data, name='original')                                                                          
mse, kl, x_rec = model(data)
plot_examples(x_rec, name='reconstructions')
