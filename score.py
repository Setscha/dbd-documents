import logging
import csv

import torch
from torch.utils.data import DataLoader, ConcatDataset
from torchvision.transforms import v2
from tqdm import tqdm

from models.mldbd.mldbd import MldbdInference, BdNet, Stage2Net
from models.unet import UNet
from utils.data_loading import BasicDataset, AugmentedDataset
from utils.metrics import f_beta_measure


def evaluate(net, dataset, device, amp):
    loader_args = dict(batch_size=1, num_workers=min(4, 16), pin_memory=True)
    dataloader = DataLoader(dataset, shuffle=False, **loader_args)

    net.eval()
    num_val_batches = len(dataloader)
    mae = 0
    f_measure = 0

    # iterate over the validation set
    with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
        for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch'):
            image, mask_true = batch['image'], batch['mask']

            # move images and labels to correct device and type
            image = image.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            mask_true = mask_true.to(device=device, dtype=torch.long)

            # predict the mask
            mask_pred = net(image)

            mask_pred = (mask_pred >= 0.5).float()
            mae += (mask_true - mask_pred).abs().mean().item()
            f_measure += f_beta_measure(mask_true, mask_pred)

    total_mae = mae / num_val_batches
    total_f_measure = f_measure / num_val_batches

    logging.info(f'Scores:\n'
                 f'MAE: {total_mae}\n'
                 f'F-measure: {total_f_measure}\n')

    net.train()
    return {
        'dataset-source': dataset.images_dir,
        'dataset-gt': dataset.mask_dir,
        'mae': total_mae,
        'f-measure': total_f_measure,
    }


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    resizeTransform = v2.Resize((320, 320))

    # Change here to adapt to your data
    # n_channels=3 for RGB images
    # n_classes is the number of probabilities you want to get per pixel
    # model = UNet(n_channels=3, n_classes=1, bilinear=False)
    model = MldbdInference()
    # model = BdNet()
    model = model.to(memory_format=torch.channels_last)

    # logging.info(f'Network:\n'
    #              f'\t{model.n_channels} input channels\n'
    #              f'\t{model.n_classes} output channels (classes)\n'
    #              f'\t{"Bilinear" if model.bilinear else "Transposed conv"} upscaling')

    model.to(device=device)
    # datasets = [
    #     BasicDataset('./data/test_data/DUT/dut500-source', './data/test_data/DUT/dut500-gt', 1),
    #     BasicDataset('./data/test_data/CUHK/xu100-source', './data/test_data/CUHK/xu100-gt', 1),
    # ]

    datasets = []
    for i in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.5, 2, 3, 5]:
        # datasets.append(BasicDataset(f'./data/cvl/testset/source/computer/{i}', f'./data/cvl/testset/gt/computer/{i}', 1))
        datasets.append(BasicDataset(f'./data/cvl/testset/source/handwritten/{i}', f'./data/cvl/testset/gt/handwritten/{i}', 1))

    datasets = [ConcatDataset(datasets)]
    datasets = [BasicDataset('./data/cvl/trainset/source/computer', './data/cvl/trainset/gt/computer', 1)]

    model_loads = [
        'checkpoints/finetune_b4_b4_1_checkpoint_epoch14.pth',
        'checkpoints/finetune_b4_b4_1_checkpoint_epoch15.pth',
        'checkpoints/finetune_b4_b4_2_checkpoint_epoch14.pth',
        'checkpoints/finetune_b4_b4_2_checkpoint_epoch15.pth',
        'checkpoints/finetune_b4_b4_3_checkpoint_epoch14.pth',
        'checkpoints/finetune_b4_b4_3_checkpoint_epoch15.pth',
    ]
    model_loads = [
        'checkpoints/finetune_b1_b4_1_checkpoint_epoch15.pth',
        'checkpoints/finetune_b1_b4_2_checkpoint_epoch15.pth',
        'checkpoints/finetune_b1_b4_3_checkpoint_epoch11.pth',
        'checkpoints/finetune_b1_b4_3_checkpoint_epoch15.pth',
    ]
    # model_loads = [
    #     'checkpoints/finetune_1_mldbd_checkpoint_epoch14_MldbdInference.pth',
    #     'checkpoints/finetune_1_mldbd_checkpoint_epoch15_MldbdInference.pth',
    #     'checkpoints/finetune_2_mldbd_checkpoint_epoch15_MldbdInference.pth',
    #     'checkpoints/finetune_3_mldbd_checkpoint_epoch15_MldbdInference.pth',
    # ]
    model_loads = [
        # 'checkpoints/2990957_checkpoint_epoch18.pth',
        # 'checkpoints/3180634_checkpoint_epoch42.pth',
        'checkpoints/3178879_checkpoint_epoch10_Stage2Net.pth',
    ]

    # model_loads = [
        # 'checkpoints/finetune_b1_b4_2_checkpoint_epoch15.pth',
        # 'checkpoints/finetune_b4_b4_3_checkpoint_epoch15.pth',
        # 'checkpoints/finetune_2_mldbd_checkpoint_epoch15_MldbdInference.pth'
    # ]

    try:
        outputs = list()
        for model_load in model_loads:
            state_dict = torch.load(model_load, map_location=device)
            model.load_state_dict(state_dict, strict=False)
            logging.info(f'Model loaded from {model_load}')
            for dataset in datasets:
                output = dict()
                output['checkpoint'] = model_load
                output.update(evaluate(model, dataset, device, amp=False))
                outputs.append(output)
        # with open('data/results_new_b1_all.csv', 'w+') as f:
        #     csv_writer = csv.DictWriter(f, fieldnames=outputs[0].keys(), delimiter=';')
        #     csv_writer.writeheader()
        #     csv_writer.writerows(outputs)
    except torch.cuda.OutOfMemoryError:
        logging.error('Detected OutOfMemoryError! '
                      'Enabling checkpointing to reduce memory usage, but this slows down training. '
                      'Consider enabling AMP (--amp) for fast and memory efficient training')
        torch.cuda.empty_cache()
