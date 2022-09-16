#!/usr/bin/env python
# coding: utf-8

"""
This file exports data loaders and constants.
train_loader_labeled, train_loader_unlabeled, valid_loader, test_loader : DataLoader
batch_size, valid_size, labeling_size
"""

import torch
from torchvision import datasets  # type: ignore
import torchvision.transforms as T  # type: ignore
import torch.utils.data as tud
import pathlib
import numpy as np
import numpy.typing as npt
import pathlib

from typing import List, Sequence, Optional, Callable, Any, Dict
from torchtyping import TensorType as TT  # type: ignore
from torchtyping import patch_typeguard   # type: ignore
from typeguard import typechecked, check_argument_types  # type: ignore

patch_typeguard()
import matplotlib.pyplot as plt  # type: ignore
from torchvision.utils import make_grid  # type: ignore


# To make it work in both Jupyter and standalone:
if "__file__" in globals():
    root = pathlib.Path(__file__).parent.resolve()
else:
    # Probably running interactively; in Jupyter, notebook path is
    # typically 'os.getcwd()', if it's not that's where we are going
    # to store the CIFAR data.
    import os
    root = pathlib.Path(os.getcwd())

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@typechecked
class LabeledCIFAR10(datasets.CIFAR10):
    def __init__(self, root : pathlib.Path,
                 indices: Sequence[int],
                 batch_size: int,
                 train : bool = True,
                 download: bool = False,
                 transform : Optional[Callable] = None):
        super().__init__(root, train = train,
                         download=download)
        # could try .to(DEVICE) maybe.
        self.data = torch.from_numpy(self.data[indices]).permute(0, 3, 1, 2).to(DEVICE)
        self.targets = torch.Tensor(self.targets)[indices].long().to(DEVICE)
        self.transform = transform
        self.batch_size = batch_size

    def __iter__(self):
        N = self.data.shape[0]
        idx = torch.randperm(N)
        data = self.transform(self.data)[idx]
        targets = self.targets[idx]

        batches_x = (data[i*self.batch_size : (i+1)*self.batch_size]
                     for i in range(N // self.batch_size))
        batches_y = (targets[i*self.batch_size :
                             (i+1)*self.batch_size] for i in range(N // self.batch_size))
        return iter(zip(batches_x, batches_y))

    def __len__(self):
        return self.data.shape[0] // self.batch_size

@typechecked
class UnlabeledCIFAR10(datasets.CIFAR10):
    def __init__(self, root : pathlib.Path,
                 indices: Sequence[int],
                 batch_size: int,
                 train : bool = True,
                 download: bool = False
                 ):
        super().__init__(root, train = train,
                         download=download)
        self.data = torch.from_numpy(self.data[indices]).permute(0, 3, 1, 2).to(DEVICE)
        self.batch_size = batch_size

    def __iter__(self):
        N = self.data.shape[0]
        idx = torch.randperm(N)
        data = self.data[idx]

        batches_x = (data[i*self.batch_size : (i+1)*self.batch_size]
                     for i in range(N // self.batch_size))
        return iter(batches_x)

    def __len__(self):
        return self.data.shape[0] // self.batch_size

class MixMatchData():
    def __init__(self, num_workers: int = 0, prefetch_factor: int = 0):
        self.transforms =  \
        {
            # For augmentation to work for mix-match, we have to keep 
            # images in 'uint8'
            "train": T.Compose([]),
            # The NN format is float images with pixels ~ N(0, 1).
            "val": T.Compose([T.ToTensor(), T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        }

        # Augmentation: AutoAugment keeps images in uints [0..255]. Next step is to 
        # convert to float [0..1] and normalize.
        self.augment = T.Compose([
            T.ConvertImageDtype(torch.float),
            T.RandomCrop(size=32,
                         padding=4,
                         padding_mode='reflect'),
            T.RandomHorizontalFlip(p=0.5),
            T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor

    @typechecked
    def get_labeled_dataloaders(self, val_prop: float = 0.2 ,
                                batch_size: int = 200,
                                augment: bool = True,
                                shuffle: bool = True) -> Dict[str, Any]:
        data_transforms = [self.transforms["train"]]
        if augment:
            data_transforms.append(self.augment)
        else:
            data_transforms += [T.ConvertImageDtype(torch.float),
                                T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        train_data = datasets.CIFAR10(root, train = True, download=True,
                    transform=T.Compose(data_transforms))

        val_data = datasets.CIFAR10(root, train = True, download=True,
                                      transform=self.transforms["val"])

        test_data = datasets.CIFAR10(root, train = False, download=True,
                    transform=T.Compose([self.transforms["val"]]))
	
        assert 0 <= val_prop < 1
        split_sizes : Sequence[int] = (
            len(train_data) * np.array([1-val_prop, val_prop])
        ).astype(int)  # type: ignore

        # Fixed seed needed for predictable results when resuming from
        # checkpoint. I think I need to set the transform before
        # splitting; otherwise it's incorrect.
        train_data, _ = tud.random_split(
            train_data,
            split_sizes,
            generator=torch.Generator().manual_seed(42))

        # This reduces train set size to just 4k:
        # train_data, _ = tud.random_split(
        #     train_data,
        #     [4000, len(train_data) - 4000],
        #     generator=torch.Generator().manual_seed(42))

        _, val_data = tud.random_split(
            val_data,
            split_sizes,
            generator=torch.Generator().manual_seed(42))

        #val_data.transform = self.transforms["val"]  # type: ignore
        to_dl = lambda ds: tud.DataLoader(ds, batch_size=batch_size,
                                          shuffle=shuffle, drop_last =
                                          True,
                                          num_workers=self.num_workers,
                                          prefetch_factor=self.prefetch_factor)

        return {"train_labeled": to_dl(train_data),
                "train_not_labeled": None,
                "val": to_dl(val_data),
                "test": to_dl(test_data)}
        

    @typechecked
    def get_dataloaders(self, labeled_prop: float = 0.5, val_prop:
                        float = 0.2 , batch_size: int = 200 , shuffle: bool = True,
                        augment: bool = True) -> Dict[str, Any]:
        data_transforms = [self.transforms["train"]]
        if augment:
            data_transforms.append(self.augment)
        else:
            data_transforms += [T.ConvertImageDtype(torch.float),
                                T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]

        assert 1 - labeled_prop - val_prop > 0
        split_proportions = np.array([labeled_prop, 1 - labeled_prop - val_prop, val_prop])
        num_labeled, num_not_labeled, num_validation = (50000*split_proportions).astype(int)

        train_data = LabeledCIFAR10(root, train=True, download=True,
                                    batch_size = batch_size,
                                    indices = list(range(num_labeled)),
                                    transform=T.Compose(data_transforms))

        train_data_not_labeled = UnlabeledCIFAR10(root, train=True,
                                                  batch_size=batch_size,
                                                  indices=[int(num_labeled) + i for i in range(num_not_labeled)],
                                                  download=True)
        test_data = datasets.CIFAR10(root, train=False, download=True,
                                     transform=self.transforms["val"])

        kwargs = {'prefetch_factor': self.prefetch_factor} if self.num_workers > 0 else {}

        to_dl = lambda ds: tud.DataLoader(ds, batch_size=batch_size,
                                          shuffle=shuffle, drop_last = True,
                                          num_workers=self.num_workers,
                                          **kwargs)

        # Ignore val data for now. It was messy to give it the val
        # transform, and not make it overlap with the 2 train sets by
        # mistake.
        return {"train_labeled": train_data,
                "train_not_labeled": train_data_not_labeled,
                "val": to_dl(test_data),
                "test": to_dl(test_data)}


    @typechecked
    def show_batch(self, dl: tud.DataLoader) -> None:
        for images, labels in dl:
            fig, ax = plt.subplots(figsize=(10,10))
            ax.imshow(make_grid(images.to(torch.device("cpu")), 10).permute(1,2,0))
            plt.show()
            break

@typechecked
def denormalize(images: TT[-1, 3, 32, 32], 
                means: Sequence[float], std_devs: Sequence[float]) -> TT[-1, 3, 32, 32]:
    means_tt : TT[1,3,1,1] = torch.tensor(means).reshape(1, 3, 1, 1)
    std_devs_tt : TT[1,3,1,1] = torch.tensor(std_devs).reshape(1, 3, 1, 1)
    return images * std_devs_tt + means_tt


@typechecked
def show_single(img: TT[3, 32, 32]) -> None:
    img = img.to(torch.device("cpu"))
    img = denormalize(img.unsqueeze(0), (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    img = img.squeeze()
    plt.imshow(img.permute(1,2,0))
    plt.show()

if __name__ == "__main__":
    data = MixMatchData(num_workers=4, prefetch_factor=2)
    dataloaders = data.get_dataloaders(labeled_prop=0.5, val_prop=0.2, batch_size=8, shuffle=False)
    classes = ['plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    data.show_batch(dataloaders["train_labeled"])
