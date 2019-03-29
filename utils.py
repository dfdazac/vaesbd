import torch
from torchvision.utils import make_grid
import matplotlib.pyplot as plt


def plot_examples(examples, name, save=False):
    min_val = torch.min(examples)
    max_val = torch.max(examples) - min_val
    scaled = (examples - min_val)/max_val
    image = make_grid(scaled)
    fig, ax = plt.subplots()
    fig.suptitle(name)
    ax.imshow(image.permute(1, 2, 0).numpy())
    if save:
        fig.savefig(name)
