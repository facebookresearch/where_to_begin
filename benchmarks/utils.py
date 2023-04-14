# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import numpy as np
import random
import os

from opacus.validators.module_validator import ModuleValidator
from torchvision import models
from torchvision.models.squeezenet import SqueezeNet, squeezenet1_0

import math
import random
from typing import Any, Dict, Generator, Iterable, Iterator, List, Tuple

import torch
from flsim.data.data_provider import IFLDataProvider, IFLUserData
from flsim.data.data_sharder import FLDataSharder
from flsim.interfaces.data_loader import IFLDataLoader
from flsim.utils.data.data_utils import batchify
from torch.utils.data import Dataset
from torchvision.datasets import VisionDataset
from tqdm import tqdm


def set_random_seed(seed_value, use_cuda: bool = True) -> None:
    np.random.seed(seed_value)  # cpu vars
    torch.manual_seed(seed_value)  # cpu  vars
    random.seed(seed_value)  # Python
    os.environ["PYTHONHASHSEED"] = str(seed_value)  # Python hash buildin
    if use_cuda:
        torch.backends.cudnn.deterministic = True  # needed
        torch.backends.cudnn.benchmark = False


def get_cnn_model(num_classes, model_type, pretrained, resnet_size=18, in_channels=3):
    """Creates global model architecture."""
    if model_type == "squeezenet1_0":
        model = (
            squeezenet1_0(pretrained=True)
            if pretrained
            else SqueezeNet(num_classes=num_classes)
        )
    elif "resnet" in model_type:
        model = Resnet(
            num_classes=num_classes, resnet_size=resnet_size, pretrained=pretrained
        )
        if in_channels == 1:
            # Change first convolution layer size to accommodate 1 channel
            model.backbone.conv1 = torch.nn.Conv2d(
                1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
            )
    elif model_type == "cnn":
        model = SimpleConvNet(in_channels=in_channels, num_classes=num_classes)
    else:
        raise ValueError(f"Model {model_type} not found")
    return model


class Resnet(nn.Module):
    """RESNET model with BatchNorm replaced with GroupNorm"""

    def __init__(self, num_classes, resnet_size, pretrained=False):
        super().__init__()

        # Retrieve resnet of appropriate size
        resnet = {
            18: models.resnet18,
            34: models.resnet34,
            50: models.resnet50,
            101: models.resnet101,
            152: models.resnet152,
        }
        assert (
            resnet_size in resnet.keys()
        ), f"Resnet size {resnet_size} is not supported!"

        self._name = f"Resnet{resnet_size}"
        self.backbone = resnet[resnet_size]()

        if pretrained:
            self.backbone.load_state_dict(
                torch.hub.load_state_dict_from_url(
                    models.resnet.model_urls[f"resnet{resnet_size}"],
                    progress=True,
                )
            )

        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)
        # Replace batch norm with group norm
        self.backbone = ModuleValidator.fix(self.backbone)

    def forward(self, x):
        return self.backbone(x)

    def name(self):
        return self._name


class SimpleConvNet(nn.Module):
    r"""
    Simple CNN model following architecture from
    https://github.com/TalwalkarLab/leaf/blob/master/models/celeba/cnn.py#L19
    and https://arxiv.org/pdf/1903.03934.pdf
    """

    def __init__(self, in_channels, num_classes, dropout_rate=0):
        super(SimpleConvNet, self).__init__()
        self.out_channels = 32
        self.stride = 1
        self.padding = 2
        self.layers = []
        in_dim = in_channels
        for _ in range(4):
            self.layers.append(
                nn.Conv2d(in_dim, self.out_channels, 3, self.stride, self.padding)
            )
            in_dim = self.out_channels
        self.layers = nn.ModuleList(self.layers)

        self.gn_relu = nn.Sequential(
            nn.GroupNorm(self.out_channels, self.out_channels, affine=True),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        num_features = (
            self.out_channels
            * (self.stride + self.padding)
            * (self.stride + self.padding)
        )
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(num_features, num_classes)

    def forward(self, x):
        for conv in self.layers:
            x = self.gn_relu(conv(x))

        x = x.view(-1, self.num_flat_features(x))
        x = self.fc(self.dropout(x))
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


class FLVisionDataLoader(IFLDataLoader):
    SEED = 2137
    random.seed(SEED)

    def __init__(
        self,
        train_dataset: VisionDataset,
        eval_dataset: VisionDataset,
        test_dataset: VisionDataset,
        sharder: FLDataSharder,
        batch_size: int,
        drop_last: bool = False,
        collate_fn=collate_fn,
    ):
        assert batch_size > 0, "Batch size should be a positive integer."
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.test_dataset = test_dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.sharder = sharder
        self.collate_fn = collate_fn

    def fl_train_set(self, **kwargs) -> Iterable[Dict[str, Generator]]:
        rank = kwargs.get("rank", 0)
        world_size = kwargs.get("world_size", 1)
        yield from self._batchify(self.train_dataset, self.drop_last, world_size, rank)

    def fl_eval_set(self, **kwargs) -> Iterable[Dict[str, Generator]]:
        yield from self._batchify(self.eval_dataset, drop_last=False)

    def fl_test_set(self, **kwargs) -> Iterable[Dict[str, Generator]]:
        yield from self._batchify(self.test_dataset, drop_last=False)

    def _batchify(
        self,
        dataset: VisionDataset,
        drop_last: bool = False,
        world_size: int = 1,
        rank: int = 0,
    ) -> Generator[Dict[str, Generator], None, None]:
        # pyre-fixme[16]: `VisionDataset` has no attribute `__iter__`.
        data_rows: List[Dict[str, Any]] = [self.collate_fn(batch) for batch in dataset]
        for index, (_, user_data) in enumerate(self.sharder.shard_rows(data_rows)):
            if index % world_size == rank and len(user_data) > 0:
                batch = {}
                keys = user_data[0].keys()
                for key in keys:
                    attribute = {
                        key: batchify(
                            [row[key] for row in user_data],
                            self.batch_size,
                            drop_last,
                        )
                    }
                    batch = {**batch, **attribute}
                yield batch


class LEAFDataLoader(IFLDataLoader):
    """Dataloader for LEAF datasets."""

    SEED = 2137
    random.seed(SEED)

    def __init__(
        self,
        train_dataset: Dataset,
        eval_dataset: Dataset,
        test_dataset: Dataset,
        batch_size: int,
        drop_last: bool = False,
    ):
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.test_dataset = test_dataset
        self.batch_size = batch_size
        self.drop_last = drop_last

    def fl_train_set(self, **kwargs) -> Iterable[Dict[str, Generator]]:
        yield from self._batchify(self.train_dataset, self.drop_last)

    def fl_eval_set(self, **kwargs) -> Iterable[Dict[str, Generator]]:
        yield from self._batchify(self.eval_dataset, drop_last=False)

    def fl_test_set(self, **kwargs) -> Iterable[Dict[str, Generator]]:
        yield from self._batchify(self.test_dataset, drop_last=False)

    @property
    def num_train_users(self):
        return len(self.train_dataset)

    @property
    def num_eval_users(self):
        return len(self.eval_dataset)

    @property
    def num_test_users(self):
        return len(self.test_dataset)

    def _batchify(
        self, dataset: Dataset, drop_last=False
    ) -> Generator[Dict[str, Generator], None, None]:
        """Group each user's data into batches.

        Args:
            dataset: Dataset object where each entry is (inputs, labels) of *one* user's
                entire data for a particular split.
            drop_last: If True, drop last remaining batch that is smaller than the
                specified batch size.

        Returns: Generator object where each entry is a dictionary that represents one
            user's data.
        """
        # pyre-fixme[16]: `Dataset` has no attribute `__iter__`.
        for one_user_inputs, one_user_labels in dataset:
            data = list(zip(one_user_inputs, one_user_labels))
            random.shuffle(data)
            one_user_inputs, one_user_labels = zip(*data)
            batch = {
                "features": batchify(one_user_inputs, self.batch_size, drop_last),
                "labels": batchify(one_user_labels, self.batch_size, drop_last),
            }
            yield batch


class LEAFUserData(IFLUserData):
    def __init__(self, user_data: Dict[str, Generator], eval_split: float):
        self._train_batches = []
        self._num_train_batches = 0
        self._num_train_examples = 0

        self._eval_batches = []
        self._num_eval_batches = 0
        self._num_eval_examples = 0

        self._eval_split = eval_split

        user_features = list(user_data["features"])
        user_labels = list(user_data["labels"])
        total = sum(len(batch) for batch in user_labels)

        for features, labels in zip(user_features, user_labels):
            if self._num_eval_examples < int(total * self._eval_split):
                self._num_eval_batches += 1
                self._num_eval_examples += LEAFUserData.get_num_examples(labels)
                self._eval_batches.append(
                    LEAFUserData.fl_training_batch(features, labels)
                )
            else:
                self._num_train_batches += 1
                self._num_train_examples += LEAFUserData.get_num_examples(labels)
                self._train_batches.append(
                    LEAFUserData.fl_training_batch(features, labels)
                )

    def train_data(self) -> Iterator[Dict[str, torch.Tensor]]:
        """Iterator for a batch of user data."""
        for batch in self._train_batches:
            yield batch

    def eval_data(self) -> Iterator[Dict[str, torch.Tensor]]:
        """Iterator for a batch of user data."""
        for batch in self._eval_batches:
            yield batch

    def num_train_batches(self):
        """Number of train batches."""
        return self._num_train_batches

    def num_eval_batches(self):
        """Number of evaluation batches."""
        return self._num_eval_batches

    def num_train_examples(self) -> int:
        """Number of train examples."""
        return self._num_train_examples

    def num_eval_examples(self):
        """Number of evaluation examples."""
        return self._num_eval_examples

    @staticmethod
    def get_num_examples(batch: List) -> int:
        """Returns the number of examples in a given `batch`."""
        return len(batch)

    @staticmethod
    def fl_training_batch(
        features: List[torch.Tensor], labels: List[float]
    ) -> Dict[str, torch.Tensor]:
        """Returns a dictionary of data in the format expected by FL dataloader."""
        return {"features": torch.stack(features), "labels": torch.Tensor(labels)}


class LEAFDataProvider(IFLDataProvider):
    """Data provider for LEAF datasets that feeds the final data for FL training."""

    def __init__(self, data_loader, eval_split=1.0):
        self.data_loader = data_loader
        self._train_users = self._create_fl_users(
            data_loader.fl_train_set(), eval_split=0.0
        )
        self._eval_users = self._create_fl_users(data_loader.fl_eval_set(), eval_split)
        self._test_users = self._create_fl_users(data_loader.fl_test_set(), eval_split)

    def train_user_ids(self) -> List[int]:
        """List of train user IDs."""
        return list(self._train_users.keys())

    def num_train_users(self) -> int:
        return len(self._train_users)

    def get_train_user(self, user_index: int) -> IFLUserData:
        if user_index in self._train_users:
            return self._train_users[user_index]
        else:
            raise IndexError(
                f"Index {user_index} is out of bound for list with len {self.num_train_users()}"
            )

    def train_users(self) -> Iterable[IFLUserData]:
        for user_data in self._train_users.values():
            yield user_data

    def eval_users(self) -> Iterable[IFLUserData]:
        for user_data in self._eval_users.values():
            yield user_data

    def test_users(self) -> Iterable[IFLUserData]:
        for user_data in self._test_users.values():
            yield user_data

    def _create_fl_users(
        self, user_data_iterator: Iterable, eval_split
    ) -> Dict[int, IFLUserData]:
        """Creates federated learning users by wrapping each user in a LEAFUserData
        class.

        Args:
            user_data_iterator: Generator for a LEAF dataset. Each entry of the iterator
                is the entire data of *one* user.

        Returns:
            Dictionary where:
                Key: user index (zero-indexed)
                Value: LEAFUserData object for this user's data
        """
        return {
            user_index: LEAFUserData(user_data, eval_split)
            for user_index, user_data in tqdm(
                enumerate(user_data_iterator), desc="Creating FL User", unit="user"
            )
        }
