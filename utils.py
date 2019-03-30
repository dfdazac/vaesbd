import torch
from torchvision.utils import make_grid
import matplotlib.backends.backend_agg as backend
from matplotlib.figure import Figure


def plot_examples(examples, name):
    clipped = torch.clamp(examples.detach(), 0, 1)
    image = make_grid(clipped)
    fig = Figure()
    canvas = backend.FigureCanvasAgg(fig)
    ax = fig.subplots()
    ax.set_title(name)
    ax.imshow(image.permute(1, 2, 0).numpy())
    canvas.print_figure(name)
