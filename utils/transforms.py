import torch
import torchvision.transforms.functional as TF
from typing import Sequence

from torchvision.transforms.v2 import Transform


class RotationChoiceTransform(Transform):
    def __init__(self, angles: Sequence[int]):
        super().__init__()
        self.angles = angles

    def __call__(self, x):
        angle = self.angles[torch.randperm(len(self.angles))[0].item()]
        return TF.rotate(x, angle, expand=True)
