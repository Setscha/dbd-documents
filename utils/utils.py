import matplotlib.pyplot as plt
import numpy as np
import torch


def plot_img_and_mask(img, mask):
    if isinstance(img, np.ndarray):
        _img = torch.from_numpy(img).contiguous()
    elif isinstance(img, torch.Tensor):
        _img = img.permute((1, 2, 0)).contiguous()
    else:
        return

    fig, ax = plt.subplots(1, 2)
    ax[0].set_title('Input image')
    ax[0].imshow(_img)
    ax[1].set_title(f'Mask')
    ax[1].imshow(mask, cmap="gray")
    plt.xticks([]), plt.yticks([])
    plt.show()
