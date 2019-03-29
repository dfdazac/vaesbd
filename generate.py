from PIL import Image, ImageDraw
import os.path as osp
import numpy as np
import torch

BG_COLOR = (255, 255, 255)
COLORS = [(0, 0, 210),
          (0, 210, 0),
          (210, 0, 0),
          (150, 150, 0),
          (150, 0, 150),
          (0, 150, 150)]

SHAPES = ['ellipse', 'rectangle', 'polygon']
num_shapes = 1

def generate_img(img_size):
    """Generate an RGN image with shapes at random positions.

    Args:
        img_size (int): side length of the image

    Returns:
        PIL.Image
    """
    img = Image.new('RGB', (img_size, img_size), color=BG_COLOR)
    drawer = ImageDraw.Draw(img)

    idx_color_shape = np.arange(len(COLORS))
    np.random.shuffle(idx_color_shape)
    size = np.random.randint(low=15, high=25, size=num_shapes)
    x0_all = np.random.uniform(0, img_size - size, size=num_shapes)
    y0_all = np.random.uniform(0, img_size - size, size=num_shapes)
    shape_idx = np.random.choice(np.arange(len(SHAPES)), size=num_shapes,
                                 replace=False)

    for i in range(num_shapes):
        x0, y0 = x0_all[i], y0_all[i]
        x1 = x0 + size[i]
        y1 = y0 + size[i]
        box = [x0, y0, x1, y1]
        shape = SHAPES[shape_idx[i]]

        if shape == 'polygon':
            box[0] = (x1 + x0) / 2
            box += [x0, y1]

        getattr(drawer, shape)(box, fill=COLORS[idx_color_shape[i]])

    return img

def generate_data(img_size, n_samples, filename):
    """Generate samples of images with shapes and save tensor to disk.
    The saved tensor has shape (n_samples, 3, img_size, img_size).

    Args:
        img_size (int): side length of the image
        n_samples (int): number of samples
        filename (str): name of file to save the data
    """
    data = torch.empty([n_samples, 3, img_size, img_size], dtype=torch.float32)
    for i in range(n_samples):
        sample = np.array(generate_img(img_size)).transpose((2, 0, 1))/255.0
        data[i] = torch.tensor(sample, dtype=torch.float32)

    torch.save(data, osp.join('data/', filename))

if __name__ == '__main__':
    np.random.seed(0)
    generate_data(img_size=64, n_samples=30000, filename='train.pt')
