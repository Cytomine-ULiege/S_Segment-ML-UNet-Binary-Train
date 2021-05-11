import os

import numpy as np
from PIL.Image import Image
import PIL
from torch.utils.data import RandomSampler

from base_dataloader import BaseDataLoader
from base_dataset import BaseDataSet


class CytomineDataset(BaseDataSet):
    def _set_files(self):
        self.image_dir = os.path.join(self.root, "images")
        self.label_dir = os.path.join(self.root, "masks")
        self.files = os.listdir(self.image_dir)

    def _load_data(self, index):
        filename = self.files[index]
        image_path = os.path.join(self.image_dir, filename)
        label_path = os.path.join(self.label_dir, filename)
        image = np.asarray(PIL.Image.open(image_path).convert('RGB'), dtype=np.float32)
        label = np.asarray(PIL.Image.open(label_path), dtype=np.int32)
        return image, label, filename

    def __init__(self, **kwargs):
        self.root = kwargs["root"]
        self.num_classes = 1
        self.palette = [0, 0, 0, 255, 255, 255]
        super(CytomineDataset, self).__init__(**kwargs)


class CytomineDataLoader(BaseDataLoader):
    def __init__(self, data_dir, batch_size, split, crop_size=None,
                 base_size=None, scale=True, num_workers=1,
                 val=False, shuffle=False, flip=False, rotate=False,
                 blur=False, augment=False, val_split=None, return_id=False, n_samples=None):

        self.MEAN = [0.45734706, 0.43338275, 0.40058118]
        self.STD = [0.23965294, 0.23532275, 0.2398498]

        kwargs = {
            'root': data_dir,
            'split': split,
            'mean': self.MEAN,
            'std': self.STD,
            'augment': augment,
            'crop_size': crop_size,
            'base_size': base_size,
            'scale': scale,
            'flip': flip,
            'blur': blur,
            'rotate': rotate,
            'return_id': return_id,
            'val': val
        }

        self.dataset = CytomineDataset(**kwargs)
        if n_samples is not None:
            sampler = RandomSampler(self.dataset, replacement=True, num_samples=n_samples)
        super(CytomineDataLoader, self).__init__(self.dataset, batch_size, shuffle,
                                                 num_workers, sampler=sampler, val_split=val_split)

