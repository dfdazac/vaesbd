import torch
from torch.utils.data import TensorDataset, DataLoader
from tensorboardX import SummaryWriter
import matplotlib.backends.backend_agg as backend
from matplotlib.figure import Figure

from model import VAE


def plot_examples(examples, name):
    clipped = torch.clamp(examples.detach(), 0, 1)
    image = make_grid(clipped)
    fig = Figure()
    canvas = backend.FigureCanvasAgg(fig)
    ax = fig.subplots()
    ax.set_title(name)
    ax.imshow(image.permute(1, 2, 0).numpy())
    canvas.print_figure(name)


device = 'cuda' if torch.cuda.is_available() else 'cpu'

train_file = 'data/train.pt'
dataset = TensorDataset(torch.load(train_file))
loader = DataLoader(dataset, batch_size=16, shuffle=True)
writer = SummaryWriter()

decoder = 'sbd'
model = VAE(im_size=64, decoder=decoder)
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

steps = 0
log = '[{:d}/{:d}] MSE: {:.6f}  KL: {:.6f}  Total: {:.6f}'
for epoch in range(100):
    print('Epoch {:d}'.format(epoch + 1))
    train_loss = 0
    train_mse = 0
    train_kl = 0
    for i, d in enumerate(loader):
        steps += 1
        batch = d[0].to(device)
        optimizer.zero_grad()
        mse_loss, kl, out = model(batch)
        loss = mse_loss + kl
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_mse += mse_loss.item()
        train_kl += kl.item()
        if (i + 1) % 200 == 0:
            train_loss /= 200
            train_mse /= 200
            train_kl /= 200
            print(log.format(i + 1, len(loader), train_mse, train_kl,
                             train_loss))
            writer.add_scalar('loss/total', train_loss, steps)
            writer.add_scalar('loss/mse', train_mse, steps)
            writer.add_scalar('loss/kl', train_kl, steps)
            train_loss = 0
            train_mse = 0
            train_kl = 0

    for name, param in model.enc_convs.named_parameters():
        writer.add_histogram(name, param.clone().cpu().data.numpy(), epoch)

    plot_examples(batch.cpu(), 'original', save=True)
    plot_examples(out.cpu().detach(), 'reconstruction', save=True)

torch.save(model.state_dict(), f'vae_{decoder}.pt')
