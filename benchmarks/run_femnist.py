#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Example Usages:
"""
import json
import random
from typing import Any, Iterator, List, Tuple

import flsim.configs  # noqa
import numpy as np
import torch
from flsim.interfaces.metrics_reporter import Channel
from flsim.utils.config_utils import maybe_parse_json_config
from flsim.utils.example_utils import (
    LEAFDataLoader,
    DataProvider,
    SimpleConvNet,
    FLModel,
    MetricsReporter,
)
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from torchvision import transforms


class FEMNISTDataset:
    IMAGE_SIZE = (28, 28)
    SEED = 4

    def __init__(
        self,
        data_root=None,
        num_users=None,
        transform=None,
        target_transform=None,
    ):
        with open(data_root) as f:
            dataset = json.load(f)

        user_ids = dataset["users"]

        random.seed(self.SEED)
        num_users = num_users if num_users is not None else len(user_ids)
        user_ids = random.sample(user_ids, min(len(user_ids), num_users))
        print(f"Creating dataset with {num_users} users")

        self.transform = transform
        self.target_transform = target_transform

        self.data = {}
        self.targets = {}
        # Populate self.data and self.targets
        for user_id in user_ids:
            if user_id in set(dataset["users"]):
                self.data[user_id] = [
                    np.array(img) for img in dataset["user_data"][user_id]["x"]
                ]
                self.targets[user_id] = list(
                    dataset["user_data"][user_id]["y"])

    def __iter__(self) -> Iterator[Tuple[List[torch.Tensor], List[Any]]]:
        for user_id in self.data.keys():
            yield self.__getitem__(user_id)

    def __getitem__(self, user_id: str) -> Tuple[List[torch.Tensor], List[Any]]:

        if user_id not in self.data or user_id not in self.targets:
            return [], []

        user_imgs, user_targets = self.data[user_id], self.targets[user_id]
        user_imgs = [
            Image.fromarray(img.reshape(FEMNISTDataset.IMAGE_SIZE)) for img in user_imgs
        ]

        user_imgs = [self.transform(img) for img in user_imgs]

        if self.target_transform is not None:
            user_targets = [self.target_transform(
                target) for target in user_targets]

        return user_imgs, user_targets

    def __len__(self) -> int:
        return len(self.data)


def build_data_provider(data_config):
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    train_dataset = FEMNISTDataset(
        data_root=data_config.train_file, transform=transform
    )
    test_dataset = FEMNISTDataset(
        data_root=data_config.test_file,
        transform=transform,
    )

    dataloader = LEAFDataLoader(
        train_dataset,
        test_dataset,
        test_dataset,
        batch_size=data_config.local_batch_size,
        drop_last=False,
    )
    return DataProvider(dataloader, data_config.eval_split)


def train(
    trainer_config,
    data_config,
    use_cuda_if_available=True,
    distributed_world_size=1,
    rank=0,
):
    metrics_reporter = MetricsReporter([Channel.TENSORBOARD, Channel.STDOUT])

    print("Created metrics reporter")
    cuda_enabled = torch.cuda.is_available() and use_cuda_if_available
    device = torch.device(
        f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    model = SimpleConvNet(1, 62)

    global_model = FLModel(model, device)
    if cuda_enabled:
        global_model.fl_cuda()

    trainer = instantiate(
        trainer_config, model=global_model, cuda_enabled=cuda_enabled)
    print(f"Created {trainer_config._target_}")

    data_provider = build_data_provider(data_config=data_config)

    final_model, train_metrics = trainer.train(
        data_provider=data_provider,
        metric_reporter=metrics_reporter,
        num_total_users=data_provider.num_users(),
        distributed_world_size=distributed_world_size,
        rank=0,
    )
    return final_model, train_metrics


def run(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    trainer_config = cfg.trainer
    data_config = cfg.data

    train(
        trainer_config=trainer_config,
        data_config=data_config,
        use_cuda_if_available=cfg.use_cuda_if_available,
        distributed_world_size=cfg.distributed_world_size,
    )


if __name__ == "__main__":
    cfg = maybe_parse_json_config()
    run(cfg)
