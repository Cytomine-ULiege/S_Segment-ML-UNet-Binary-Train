import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from cytomine.utilities.software import parse_domain_list
from rasterio.features import geometry_mask
from rasterio.transform import IDENTITY
from shapely import wkt
from shapely.affinity import affine_transform


from cytomine import CytomineJob
from cytomine.models import ImageInstanceCollection, AnnotationCollection, AttachedFile
from torch.optim import SGD
from torch.utils.data import DataLoader, RandomSampler

from cytomine_loader import CytomineDataLoader
from unet_model import UNet


def get_instance(module, name, config, *args):
    # GET THE CORRESPONDING CLASS / FCT
    return getattr(module, config[name]['type'])(*args, **config[name]['args'])


class Monitor(object):
    def __init__(self, job, iterable, start=0, end=100, period=None, prefix=None):
        self._job = job
        self._start = start
        self._end = end
        self._update_period = period
        self._iterable = iterable
        self._prefix = prefix

    def update(self, *args, **kwargs):
        return self._job.job.update(*args, **kwargs)

    def _get_period(self, n_iter):
        """Return integer period given a maximum number of iteration """
        if self._update_period is None:
            return None
        if isinstance(self._update_period, float):
            return max(int(self._update_period * n_iter), 1)
        return self._update_period

    def _relative_progress(self, ratio):
        return int(self._start + (self._end - self._start) * ratio)

    def __iter__(self):
        total = len(self)
        for i, v in enumerate(self._iterable):
            period = self._get_period(total)
            if period is None or i % period == 0:
                statusComment = "{} ({}/{}).".format(self._prefix, i + 1, len(self))
                relative_progress = self._relative_progress(i / float(total))
                self._job.job.update(progress=relative_progress, statusComment=statusComment)
            yield v

    def __len__(self):
        return len(list(self._iterable))



def main(argv):
    with CytomineJob.from_cli(argv) as cj:
        working_path = str(Path.home())
        data_path = os.path.join(working_path, "data")
        images_dir = os.path.join(data_path, "images")
        mask_dir = os.path.join(data_path, "masks")
        model_path = os.path.join(working_path, "model")

        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(mask_dir, exist_ok=True)
        os.makedirs(model_path, exist_ok=True)

        foreground_terms = parse_domain_list(cj.parameters.cytomine_foreground_terms)
        annotations = AnnotationCollection(terms=foreground_terms, project=[cj.project.id], showWKT=True, showMeta=True).fetch()

        annot_per_image = defaultdict(list)
        for annot in annotations:
            annot_per_image[annot.image].append(wkt.loads(annot.location))

        images = ImageInstanceCollection().fetch_with_filter("project", cj.parameters.cytomine_id_project)
        filepaths = list()
        for img in images:
            filepath = os.path.join(images_dir, img.originalFilename)
            filepaths.append(filepath)
            img.download(filepath, override=False)
            pil_image = Image.open(filepath)
            geoms = [affine_transform(g, [1, 0, 0, -1, 0, img.height]) for g in annot_per_image[img.id]]
            shape = (pil_image.height, pil_image.width)
            drawn = Image.fromarray(geometry_mask(geoms, shape, IDENTITY, invert=True))
            drawn.save(os.path.join(mask_dir, img.originalFilename))

        batch_size = 8
        device = torch.device("cpu")
        net = UNet(3, 2)
        net.to(device)
        net.train()

        optimizer = SGD(net.parameters(),
                        lr=cj.parameters.learning_rate,
                        momentum=cj.parameters.momentum,
                        weight_decay=cj.parameters.weight_decay)

        loss_fn = torch.nn.CrossEntropyLoss()

        loader = CytomineDataLoader(data_path, batch_size, "", crop_size=384, base_size=400, augment=True,
                                    shuffle=False, scale=True, flip=True, blur=True, num_workers=cj.parameters.n_jobs,
                                    n_samples=cj.parameters.n_epochs * len(filepaths))

        for i, (x, y_true) in cj.monitor(enumerate(loader), period=1, prefix="training"):
            y = net.forward(x)
            optimizer.zero_grad()
            loss = loss_fn(y, y_true)
            loss.backward()
            optimizer.step()
            print("{} - {}".format(i, loss.detach().cpu().numpy()))

        model_filepath = os.path.join(model_path, "model.pth")
        torch.save(net.state_dict(), model_filepath)
        AttachedFile(
            cj.job,
            domainIdent=cj.job.id,
            filename=model_filepath,
            domainClassName="be.cytomine.processing.Job"
        ).upload()


if __name__ == "__main__":
    main(sys.argv[1:])

