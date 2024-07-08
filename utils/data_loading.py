import logging
from os import listdir
from os.path import splitext, isfile, join
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms.v2 import Transform


def load_image(filename):
    ext = splitext(filename)[1]
    if ext == '.npy':
        return Image.fromarray(np.load(filename))
    elif ext in ['.pt', '.pth']:
        return Image.fromarray(torch.load(filename).numpy())
    else:
        image = Image.open(filename)
        return image.convert('RGB') if image.mode == 'RGBA' else image


def calculate_scaled_size(size: tuple[int, int], scale: float, max_size: tuple[int, int] | None = None) -> tuple[int, int]:
    w, h = size
    new_w, new_h = int(scale * w), int(scale * h)
    assert new_w > 0 and new_h > 0, 'Scale is too small, resized images would have no pixel'
    if max_size is None:
        return new_w, new_h

    max_w, max_h = max_size
    if new_w > new_h:
        scale = max_w / new_w
        if scale < 1:
            new_w, new_h = int(scale * new_w), int(scale * new_h)
    else:
        scale = max_h / new_h
        if scale < 1:
            new_w, new_h = int(scale * new_w), int(scale * new_h)
    return new_w, new_h


class BasicDataset(Dataset):
    def __init__(self, images_dir: str, mask_dir: str, scale: float, mask_suffix: str = ''):
        self.images_dir = Path(images_dir)
        self.mask_dir = Path(mask_dir)
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        self.scale = scale
        self.mask_suffix = mask_suffix

        self.ids = [splitext(file)[0] for file in listdir(images_dir) if
                    isfile(join(images_dir, file)) and not file.startswith('.')]
        if not self.ids:
            raise RuntimeError(f'No input file found in {images_dir}, make sure you put your images there')

        logging.info(f'Creating dataset with {len(self)} examples')

    def __len__(self):
        return len(self.ids)

    @staticmethod
    def preprocess(pil_img, scale, is_mask):
        w, h = pil_img.size
        new_w, new_h = calculate_scaled_size(pil_img.size, scale)
        if new_w != w or new_h != h:
            pil_img = pil_img.resize((new_w, new_h), resample=Image.NEAREST if is_mask else Image.BICUBIC)
        img = np.asarray(pil_img)

        if is_mask:
            if img.ndim == 2:
                mask = img.copy()
            else:
                mask = np.ndarray([img[0], img[1]])

            mask = mask / 255.0
            mask[mask < 0.5] = 0
            mask[mask >= 0.5] = 1
            return mask

        else:
            if img.ndim == 2:
                img = img[np.newaxis, ...]
            else:
                img = img.transpose((2, 0, 1))

            if (img > 1).any():
                img = img / 255.0

            return img

    def __getitem__(self, idx):
        name = self.ids[idx]
        mask_file = list(self.mask_dir.glob(name + self.mask_suffix + '.*'))
        img_file = list(self.images_dir.glob(name + '.*'))

        assert len(img_file) == 1, f'Either no image or multiple images found for the ID {name}: {img_file}'
        assert len(mask_file) == 1, f'Either no mask or multiple masks found for the ID {name}: {mask_file}'
        mask = load_image(mask_file[0])
        img = load_image(img_file[0])

        assert img.size == mask.size, \
            f'Image and mask {name} should be the same size, but are {img.size} and {mask.size}'

        img = self.preprocess(img, self.scale, is_mask=False)
        mask = self.preprocess(mask, self.scale, is_mask=True)

        return {
            'image': torch.as_tensor(img.copy()).float().contiguous(),
            'mask': torch.as_tensor(mask.copy()).float().contiguous()
        }


class AugmentedDataset(Dataset):
    def __init__(self, images_dir: str, mask_dir: str, scale: float, transforms: list[Transform] = None,
                 mask_suffix: str = ''):
        """
        Create a Dataset with augmented duplications of images and masks.
        :param images_dir: The path to the image files
        :param mask_dir: The path to the mask files
        :param scale: The scale factor of the image as a preprocessing step (before transforms are applied)
        :param transforms: a list of transformations to apply to the dataset. Each transformation is applied
        sequentially to the dataset, resulting in a total amount of len(transforms) * amount of images
        :param mask_suffix: The suffix of mask files
        """
        self.images_dir = Path(images_dir)
        self.mask_dir = Path(mask_dir)
        self.transforms = transforms
        self.mask_suffix = mask_suffix

        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        self.scale = scale

        self.ids = [splitext(file)[0] for file in listdir(images_dir) if
                    isfile(join(images_dir, file)) and not file.startswith('.')]
        if not self.ids:
            raise RuntimeError(f'No input file found in {images_dir}, make sure you put your images there')

        logging.info(f'Creating dataset with {len(self)} examples')

    def __len__(self):
        if self.transforms is not None and len(self.transforms) > 0:
            return len(self.ids) * len(self.transforms)
        else:
            return len(self.ids)

    @staticmethod
    def preprocess(pil_img: Image, scale: float, is_mask: bool):
        w, h = pil_img.size
        new_w, new_h = calculate_scaled_size(pil_img.size, scale)
        if new_w != w or new_h != h:
            pil_img = pil_img.resize((new_w, new_h), resample=Image.NEAREST if is_mask else Image.BICUBIC)
        img = np.asarray(pil_img)

        if is_mask:
            if img.ndim == 2:
                mask = img.copy()
            else:
                mask = np.ndarray([img[0], img[1]])

            mask = mask / 255.0
            mask[mask < 0.5] = 0
            mask[mask >= 0.5] = 1
            return mask

        else:
            if img.ndim == 2:
                img = img[np.newaxis, ...]
            else:
                img = img.transpose((2, 0, 1))

            if (img > 1).any():
                img = img / 255.0

            return img

    def __getitem__(self, idx):
        name = self.ids[idx % len(self.ids)]
        mask_file = list(self.mask_dir.glob(name + self.mask_suffix + '.*'))
        img_file = list(self.images_dir.glob(name + '.*'))

        assert len(img_file) == 1, f'Either no image or multiple images found for the ID {name}: {img_file}'
        assert len(mask_file) == 1, f'Either no mask or multiple masks found for the ID {name}: {mask_file}'
        mask = load_image(mask_file[0])
        img = load_image(img_file[0])

        assert img.size == mask.size, \
            f'Image and mask {name} should be the same size, but are {img.size} and {mask.size}'

        img = self.preprocess(img, self.scale, is_mask=False)
        mask = self.preprocess(mask, self.scale, is_mask=True)

        if self.transforms is not None and len(self.transforms) > 0:
            transform = self.transforms[idx // len(self.ids)]
            state = torch.get_rng_state()
            transformed_img = transform(torch.from_numpy(img.copy()))
            torch.set_rng_state(state)
            transformed_mask = transform(torch.from_numpy(mask.copy()).repeat((3, 1, 1)))
            return {
                'image': transformed_img.float().contiguous(),
                'mask': transformed_mask[0, :, :].float().contiguous()
            }
        else:
            return {
                'image': torch.as_tensor(img.copy()).float().contiguous(),
                'mask': torch.as_tensor(mask.copy()).float().contiguous()
            }


class SameClassDataset(Dataset):
    def __init__(self, images_dir: str, mask_val: int, scale: float, max_size: tuple[int, int]):
        """
        Create a Dataset of images all having the same class for all pixels.
        :param images_dir: The path to the image files
        :param mask_val: The mask value
        :param max_size: The maximum size an image is allowed to have in (width, height)
        """
        self.images_dir = Path(images_dir)
        self.mask_val = mask_val
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        self.scale = scale
        self.max_size = max_size

        self.ids = [splitext(file)[0] for file in listdir(images_dir) if
                    isfile(join(images_dir, file)) and not file.startswith('.')]
        if not self.ids:
            raise RuntimeError(f'No input file found in {images_dir}, make sure you put your images there')

        logging.info(f'Creating dataset with {len(self)} examples')

    def __len__(self):
        return len(self.ids)

    def preprocess(self, pil_img):
        w, h = pil_img.size
        new_w, new_h = calculate_scaled_size(pil_img.size, self.scale, self.max_size)
        if new_w != w or new_h != h:
            pil_img = pil_img.resize((new_w, new_h), resample=Image.BICUBIC)
        img = np.asarray(pil_img)

        if img.ndim == 2:
            img = img[np.newaxis, ...]
        else:
            img = img.transpose((2, 0, 1))

        if (img > 1).any():
            img = img / 255.0

        return img

    def __getitem__(self, idx):
        name = self.ids[idx]
        img_file = list(self.images_dir.glob(name + '.*'))

        assert len(img_file) == 1, f'Either no image or multiple images found for the ID {name}: {img_file}'
        img = load_image(img_file[0])

        img = self.preprocess(img)

        return {
            'image': torch.as_tensor(img.copy()).float().contiguous(),
            'mask': torch.full((img.shape[1], img.shape[2]), self.mask_val).float().contiguous()
        }


class SameClassAugmentedDataset(SameClassDataset):
    def __init__(self, images_dir: str, mask_val: int, scale: float, max_size: tuple[int, int], transforms: list[Transform] = None):
        """
        Create a Dataset of images all having the same class for all pixels.
        :param images_dir: The path to the image files
        :param mask_val: The mask value
        :param max_size: The maximum size an image is allowed to have in (width, height)
        :param transforms: a list of transformations to apply to the dataset. Each transformation is applied
        sequentially to the dataset, resulting in a total amount of len(transforms) * amount of images
        """
        self.transforms = transforms
        super().__init__(images_dir, mask_val, scale, max_size)

    def __len__(self):
        if self.transforms is not None and len(self.transforms) > 0:
            return len(self.ids) * len(self.transforms)
        else:
            return len(self.ids)

    def __getitem__(self, idx):
        data = SameClassDataset.__getitem__(self, idx % len(self.ids))

        if self.transforms is not None and len(self.transforms) > 0:
            transform = self.transforms[idx // len(self.ids)]
            transformed_img = transform(data['image'])
            return {
                'image': transformed_img.float().contiguous(),
                'mask': torch.full((transformed_img.shape[1], transformed_img.shape[2]), self.mask_val).float().contiguous()
            }
        else:
            return data
