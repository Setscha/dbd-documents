import argparse
import logging
import os
import random
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from pathlib import Path

from PIL.Image import Image
from matplotlib import pyplot as plt
from torch import optim, Tensor
from torch.utils.data import DataLoader, random_split, ConcatDataset
from torchvision.transforms import v2
from tqdm import tqdm

import utils.utils
import wandb
from evaluate import evaluate
from models.mldbd.mldbd import BdNet, CdNet
from models.unet import UNet
from utils.data_loading import BasicDataset, AugmentedDataset, SameClassDataset, SameClassAugmentedDataset
from utils.metrics import dice_loss, binary_dice_loss, cyclic_lr_scale_fn
from utils.mldbd.losses import loss_bd, loss_cd, loss_c, loss_b, loss_d
from utils.transforms import RotationChoiceTransform

dir_checkpoint = Path(f'{os.environ["DATA"]}/dbd-documents/checkpoints/')
# dir_checkpoint = Path('./checkpoints/')

def show_dbd_edge_gt(dbd_pred: Tensor, edge_pred: Tensor, gt: Tensor):
    fig, axs = plt.subplots(1, 3)
    axs[0].imshow(dbd_pred.cpu().detach().squeeze(0).permute(1, 0).numpy(), cmap='gray', vmin=0, vmax=1)
    axs[0].set_title('DBD Image')
    axs[1].imshow(edge_pred.cpu().detach().squeeze(0).permute(1, 0).numpy(), cmap='gray', vmin=0, vmax=1)
    axs[1].set_title('Edge Image')
    axs[2].imshow(gt.cpu().detach().squeeze(0).permute(1, 0).numpy(), cmap='gray', vmin=0, vmax=1)
    axs[2].set_title('GT Image')
    plt.show()

def show_dbd_class_gt(dbd_pred: Tensor, class_pred: Tensor, gt: Tensor):
    fig, axs = plt.subplots(1, 3)
    axs[0].imshow(dbd_pred.cpu().detach().squeeze(0).permute(1, 0).numpy(), cmap='gray', vmin=0, vmax=1)
    axs[0].set_title('DBD Image')
    axs[1].set_title('Class: ' + str(class_pred))
    axs[2].imshow(gt.cpu().detach().squeeze(0).permute(1, 0).numpy(), cmap='gray', vmin=0, vmax=1)
    axs[2].set_title('GT Image')
    plt.show()


def train_model(
    model,
    device,
    epochs: int = 5,
    batch_size: int = 1,
    learning_rate: float = 1e-5,
    val_percent: float = 0.1,
    save_checkpoint: bool = True,
    img_scale: float = 0.5,
    amp: bool = False,
    weight_decay: float = 1e-8,
    momentum: float = 0.999,
    gradient_clipping: float = 1.0,
):
    # 1. Create dataset
    transforms = [
        v2.Compose([
            v2.Resize((320, 320)),
        ]),
        v2.Compose([
            v2.Resize((320, 320)),
            v2.RandomHorizontalFlip(p=1),
        ]),
        v2.Compose([
            v2.Resize((320, 320)),
            v2.RandomVerticalFlip(p=1),
        ]),
        v2.Compose([
            v2.Resize((320, 320)),
            RotationChoiceTransform([90, 180, 270]),
        ]),
    ]
    datasets = [
        # BasicDataset('./data/train_data/1204source', './data/train_data/1204gt', img_scale),
        # AugmentedDataset('./data/train_data/1204source', './data/train_data/1204gt', img_scale, transforms),
        # SameClassDataset('./data/train_data/FCFB/FC', 1, img_scale, (360, 360)),
        SameClassAugmentedDataset('./data/train_data/FCFB/FC', 1, img_scale, (9999, 9999), transforms),
        # SameClassDataset('./data/train_data/FCFB/FB', 0, img_scale, (360, 360)),
        SameClassAugmentedDataset('./data/train_data/FCFB/FB', 0, img_scale, (9999, 9999), transforms)
    ]

    # 2. Split into train / validation partitions
    n_train = 0
    n_val = 0
    train_sets = list()
    val_sets = list()
    for dataset in datasets:
        dataset_n_val = int(len(dataset) * val_percent)
        dataset_n_train = len(dataset) - dataset_n_val
        n_val += dataset_n_val
        n_train += dataset_n_train
        train_set, val_set = random_split(dataset, [dataset_n_train, dataset_n_val], generator=torch.Generator().manual_seed(0))
        train_sets.append(train_set)
        val_sets.append(val_set)

    train_set = ConcatDataset(train_sets)
    val_set = ConcatDataset(val_sets)

    # 3. Create data loaders
    workers = None
    if hasattr(os, 'sched_getaffinity'):
        try:
            workers = len(os.sched_getaffinity(0))
        except Exception:
            pass
    if workers is None:
        cpu_count = os.cpu_count()
        if cpu_count is not None:
            workers = cpu_count

    loader_args = dict(batch_size=batch_size, num_workers=min(workers, 8), pin_memory=True)
    val_loader_args = dict(batch_size=1, num_workers=loader_args['num_workers'], pin_memory=loader_args['pin_memory'])
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **val_loader_args)

    # 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ConstantLR(optimizer)
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    global_step = 0

    # (Initialize logging)
    experiment = wandb.init(project='DBD-Documents-MLDBD', resume='allow', anonymous='allow', mode='online')
    experiment.config.update(
        dict(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate, jobid=os.environ['SLURM_JOBID'],
             val_percent=val_percent, save_checkpoint=save_checkpoint, img_scale=img_scale, amp=amp,
             scheduler=type(scheduler).__name__, optimizer=type(optimizer).__name__, model=type(model).__name__)
    )

    logging.info(f'''Starting training:
            Epochs:          {epochs}
            Batch size:      {batch_size}
            Learning rate:   {learning_rate}
            Training size:   {n_train}
            Validation size: {n_val}
            Checkpoints:     {save_checkpoint}
            Device:          {device.type}
            Mixed Precision: {amp}
        ''')

    # 5. Begin training
    best_score = None
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                images, true_masks = batch['image'], batch['mask']

                images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
                true_masks = true_masks.to(device=device, dtype=torch.long)

                with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
                    # dbd_pred, boundary_pred = model(images)
                    dbd_pred, class1 = model(images)
                    # loss = loss_bd(dbd_pred, boundary_pred, true_masks.float())
                    loss = loss_cd(dbd_pred, class1, true_masks.float())
                    # loss = loss_d(dbd_pred, true_masks.float())

                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                grad_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
                grad_scaler.step(optimizer)
                grad_scaler.update()
                scheduler.step()

                pbar.update(images.shape[0])
                global_step += 1
                epoch_loss += loss.item()
                experiment.log({
                    'learning rate': optimizer.param_groups[0]['lr'],
                    'train loss': loss.item(),
                    'step': global_step,
                    'epoch': epoch
                })
                pbar.set_postfix(**{'loss (batch)': f'{loss.item()}'})

                # Debug output
                # if global_step % (n_train // batch_size // 3) == 0 or global_step == 1:
                    # show_dbd_edge_gt(dbd_pred[0], boundary_pred[0], true_masks[0])
                    # show_dbd_class_gt(dbd_pred[0], class1[0], true_masks[0])

                # Evaluation round
                division_step = (n_train // (1 * batch_size))
                if division_step > 0:
                    if global_step % division_step == 0:
                        histograms = {}
                        for tag, value in model.named_parameters():
                            tag = tag.replace('/', '.')
                            if not (torch.isinf(value) | torch.isnan(value)).any():
                                histograms['Weights/' + tag] = wandb.Histogram(value.data.cpu())
                            if value.grad is not None and not (torch.isinf(value.grad) | torch.isnan(value.grad)).any():
                                histograms['Gradients/' + tag] = wandb.Histogram(value.grad.data.cpu())

                        val_score = evaluate(model, val_loader, device, amp)
                        # scheduler.step(val_score)

                        logging.info('Validation Dice score: {}'.format(val_score))
                        try:

                            experiment.log({
                                'validation Dice': val_score,
                                'images': wandb.Image(images[0].cpu()),
                                'masks': {
                                    'true': wandb.Image(TF.to_pil_image(true_masks[0].float().cpu()), 'L'),
                                    'pred': wandb.Image(TF.to_pil_image((dbd_pred[0] >= 0.5).float().cpu()), 'L'),
                                },
                                'step': global_step,
                                'epoch': epoch,
                                **histograms
                            })
                        except:
                            pass

        if save_checkpoint or epoch == epochs or best_score is None or val_score < best_score:
            if best_score is None or val_score < best_score:
                best_score = val_score
            Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
            state_dict = model.state_dict()
            torch.save(state_dict, str(dir_checkpoint / f'{os.environ["SLURM_JOBID"]}_checkpoint_epoch{epoch}_{type(model).__name__}.pth'))
            logging.info(f'Checkpoint {epoch} saved!')


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    model = CdNet()
    model = model.to(memory_format=torch.channels_last)

    logging.info(f'Network: {type(model).__name__}')

    model.to(device=device)
    try:
        train_model(
            model=model,
            epochs=300,
            batch_size=4,
            learning_rate=0.0001,
            device=device,
            val_percent=0.1,
            momentum=0.9,
            amp=False,
            save_checkpoint=False
        )
    except torch.cuda.OutOfMemoryError:
        logging.error('Detected OutOfMemoryError! '
                      'Enabling checkpointing to reduce memory usage, but this slows down training. '
                      'Consider enabling AMP (--amp) for fast and memory efficient training')
        torch.cuda.empty_cache()
        model.use_checkpointing()

