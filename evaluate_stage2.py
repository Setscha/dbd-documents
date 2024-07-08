import random

import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from torch import Tensor
from tqdm import tqdm

from utils.metrics import f_beta_measure, multiclass_dice_coeff


def show_dbd_edge_gt(dbd_pred: Tensor, edge_pred: Tensor, gt: Tensor):
    fig, axs = plt.subplots(1, 3)
    axs[0].imshow(dbd_pred.cpu().detach().squeeze(0).permute(1, 0).numpy(), cmap='gray', vmin=0, vmax=1)
    axs[0].set_title('DBD Image')
    axs[1].imshow(edge_pred.cpu().detach().squeeze(0).permute(1, 0).numpy(), cmap='gray', vmin=0, vmax=1)
    axs[1].set_title('Edge Image')
    axs[2].imshow(gt.cpu().detach().permute(1, 0).numpy(), cmap='gray', vmin=0, vmax=1)
    axs[2].set_title('GT Image')
    plt.title('AS')
    plt.show()

@torch.inference_mode()
def evaluate_stage2(net, dataloader, t3, t4, device, amp):
    net.eval()
    num_val_batches = len(dataloader)
    dice_score = 0

    # iterate over the validation set
    with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
        for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch'):
            image, mask_true = batch['image'], batch['mask']

            # move images and labels to correct device and type
            image = image.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            mask_true = mask_true.to(device=device, dtype=torch.long)

            # predict the mask
            mask_pred, _, _, _ = net(image,
                                     t3[random.randint(0, len(t3) - 1)]['image'].unsqueeze(0).to(device),
                                     t4[random.randint(0, len(t4) - 1)]['image'].unsqueeze(0).to(device),
                                     mask_true)
            # show_dbd_edge_gt(mask_pred[0], _[0], mask_true[0])

            # if net.n_classes == 1:
            assert mask_true.min() >= 0 and mask_true.max() <= 1, 'True mask indices should be in [0, 1]'
            mask_pred = (mask_pred >= 0.5).float()
            # compute the Dice score
            # dice_score += f_beta_measure(mask_pred.squeeze(1), mask_true)
            # dice_score += mae(mask_pred.squeeze(1), mask_true)
            dice_score += (mask_true - mask_pred).abs().mean().item()
            # else:
            #     assert mask_true.min() >= 0 and mask_true.max() < net.n_classes, 'True mask indices should be in [0, n_classes['
                # convert to one-hot format
                # mask_true = F.one_hot(mask_true, net.n_classes).permute(0, 3, 1, 2).float()
                # mask_pred = F.one_hot(mask_pred.argmax(dim=1), net.n_classes).permute(0, 3, 1, 2).float()
                # compute the Dice score, ignoring background
                # dice_score += multiclass_dice_coeff(mask_pred[:, 1:], mask_true[:, 1:], reduce_batch_first=False)

    net.train()
    return dice_score / max(num_val_batches, 1)
