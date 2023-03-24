#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""This file runs cifar10 and cifar100 partioned using dirichlet.

The dataset must be created beforehand using this notebook https://colab.research.google.com/drive/1GeQOB2VGaj4qPXpL4j2ojBiNKEQ3CAsB?usp=sharing
"""
from typing import Dict, NamedTuple

import flsim.fb.configs  # noqa
import hydra  # @manual
import torch

from flsim.baselines.data.data_providers import LEAFDataLoader, LEAFDataProvider
from .utils import get_cnn_model, set_random_seed


from flsim.metrics_reporter.tensorboard_metrics_reporter import FLMetricsReporter
from flsim.interfaces.metrics_reporter import Channel
from flsim.utils.config_utils import maybe_parse_json_config
from hydra.utils import instantiate  # @manual
from omegaconf import DictConfig, OmegaConf
from torchvision import transforms
from torchvision.datasets import CIFAR10
from flsim.interfaces.metrics_reporter import Channel, TrainingStage

from typing import Any, Dict, List, Optional
from flsim.common.timeline import Timeline

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from flsim.baselines.data.data_providers import LEAFUserData
from flsim.interfaces.model import IFLModel
from flsim.utils.simple_batch_metrics import FLBatchMetrics


class FLModel(IFLModel):
    def __init__(self, model: nn.Module, device: Optional[str] = None):
        self.model = model
        self.device = device

    def fl_forward(self, batch) -> FLBatchMetrics:
        features = batch["features"]  # [B, C, 28, 28]
        batch_label = batch["labels"]
        stacked_label = batch_label.view(-1).long().clone().detach()
        if self.device is not None:
            features = features.to(self.device)

        output = self.model(features)

        if self.device is not None:
            output, batch_label, stacked_label = (
                output.to(self.device),
                batch_label.to(self.device),
                stacked_label.to(self.device),
            )

        loss = F.cross_entropy(output, stacked_label)
        num_examples = self.get_num_examples(batch)
        output = output.detach().cpu()
        stacked_label = stacked_label.detach().cpu()
        del features
        return FLBatchMetrics(
            loss=loss,
            num_examples=num_examples,
            predictions=output,
            targets=stacked_label,
            model_inputs=[],
        )

    def fl_create_training_batch(self, **kwargs):
        features = kwargs.get("features", None)
        labels = kwargs.get("labels", None)
        return LEAFUserData.fl_training_batch(features, labels)

    def fl_get_module(self) -> nn.Module:
        return self.model

    def fl_cuda(self) -> None:
        self.model = self.model.to(self.device)  # pyre-ignore

    def get_eval_metrics(self, batch) -> FLBatchMetrics:
        with torch.no_grad():
            return self.fl_forward(batch)

    def get_num_examples(self, batch) -> int:
        return LEAFUserData.get_num_examples(batch["labels"])


class CVMetricsReporter(FLMetricsReporter):
    ACCURACY = "Accuracy"
    ROUND_TO_TARGET = "round_to_target"

    def __init__(
        self,
        channels: List[Channel],
        log_dir: Optional[str] = None,
    ):
        super().__init__(channels, log_dir)
        self.best_score: Dict[str, Any] = {}
        self._set_summary_writer(log_dir=log_dir)
        self._round_to_target = float(1e10)

    def compare_metrics(self, eval_metrics, best_metrics):
        print(
            f"Current eval accuracy: {eval_metrics}%, Best so far: {best_metrics}%")
        if best_metrics is None:
            return True

        current_accuracy = eval_metrics.get(self.ACCURACY, float("-inf"))
        best_accuracy = best_metrics.get(self.ACCURACY, float("-inf"))
        self.best_score[self.ACCURACY] = max(best_accuracy, current_accuracy)
        return current_accuracy > best_accuracy

    def compute_scores(self) -> Dict[str, Any]:
        # compute accuracy
        correct = torch.Tensor([0])
        for i in range(len(self.predictions_list)):
            all_preds = self.predictions_list[i]
            pred = all_preds.data.max(1, keepdim=True)[1]

            assert pred.device == self.targets_list[i].device, (
                f"Pred and targets moved to different devices: "
                f"pred >> {pred.device} vs. targets >> {self.targets_list[i].device}"
            )
            if i == 0:
                correct = correct.to(pred.device)

            correct += pred.eq(self.targets_list[i].data.view_as(pred)).sum()

        # total number of data
        total = sum(len(batch_targets) for batch_targets in self.targets_list)

        accuracy = 100.0 * correct.item() / total
        best_accuracy = self.best_score.get(self.ACCURACY, float("-inf"))
        self.best_score[self.ACCURACY] = max(best_accuracy, accuracy)
        return {self.ACCURACY: accuracy, self.ROUND_TO_TARGET: self._round_to_target}

    def create_eval_metrics(
        self, scores: Dict[str, Any], total_loss: float, **kwargs
    ) -> Any:
        timeline: Timeline = kwargs.get("timeline", Timeline(global_round=1))
        stage: TrainingStage = kwargs.get("stage", None)
        accuracy = scores[self.ACCURACY]
        if stage in [
            TrainingStage.EVAL,
            TrainingStage.TEST,
        ] and self._stats.update_and_check_target(accuracy):
            self._round_to_target = min(
                timeline.global_round_num(), self._round_to_target
            )
        return {
            self.ACCURACY: accuracy,
            self.ROUND_TO_TARGET: self._round_to_target,
        }

    def get_best_score(self):
        return self.best_score


class CIFAROutput(NamedTuple):
    log_dir: str
    eval_scores: Dict[str, float]
    test_scores: Dict[str, float]


def build_data_provider(
    data_config, drop_last: bool = False, use_cifar100: bool = False
):

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
        ]
    )

    with open(data_config.train_file, "r+") as f:
        train_dataset = torch.load(f)
    with open(data_config.eval_file, "r+") as f:
        eval_dataset = torch.load(f)

    test_dataset = CIFAR10(train=False, download=True, transform=transform)

    data_loader = LEAFDataLoader(
        train_dataset,
        eval_dataset,
        # pyre-fixme[6]: Expected `Dataset[typing.Any]` for 3rd param but got
        #  `List[typing.List[typing.Any]]`.
        # pyre-fixme[32]: Variable argument must be an iterable.
        [list(zip(*test_dataset))],
        data_config.local_batch_size,
    )
    data_provider = LEAFDataProvider(data_loader)
    print(f"Clients in total: {data_provider.num_train_users()}")
    return data_provider


def train(
    trainer_config,
    model_config,
    data_config,
    use_cuda_if_available: bool = True,
    world_size: int = 1,
    rank: int = 0,
) -> CIFAROutput:

    cuda_enabled = torch.cuda.is_available() and use_cuda_if_available
    device = torch.device(f"cuda:{rank}" if cuda_enabled else "cpu")
    set_random_seed(model_config.seed, cuda_enabled)
    print(f"Training launched on device: {device}")

    data_provider = build_data_provider(data_config)

    metrics_reporter = CVMetricsReporter(
        [Channel.TENSORBOARD, Channel.STDOUT],
        target_eval=model_config.target_eval,
        window_size=model_config.window_size,
        average_type=model_config.average_type,
    )
    print("Created metrics reporter")

    cuda_enabled = torch.cuda.is_available() and use_cuda_if_available
    device = torch.device(f"cuda:{rank}" if cuda_enabled else "cpu")
    print(f"Training launched on device: {device}")

    num_classes = 10

    model = get_cnn_model(
        num_classes=num_classes,
        model_type=model_config.model_type,
        pretrained=model_config.pretrained,
    )

    # pyre-fixme[6]: Expected `Optional[str]` for 2nd param but got `device`.
    global_model = FLModel(model, device)
    if cuda_enabled:
        global_model.fl_cuda()

    trainer = instantiate(
        trainer_config, model=global_model, cuda_enabled=cuda_enabled)
    print(f"Created {trainer_config._target_}")

    final_model, eval_score = trainer.train(
        data_provider=data_provider,
        metrics_reporter=metrics_reporter,
        num_total_users=data_provider.num_train_users(),
        distributed_world_size=world_size,
        rank=rank,
    )
    test_metric = trainer.test(
        data_provider=data_provider,
        metrics_reporter=CVMetricsReporter([Channel.STDOUT]),
    )
    return CIFAROutput(
        log_dir=metrics_reporter.writer.log_dir,
        eval_scores=eval_score,
        test_scores=test_metric,
    )


@hydra.main(config_path=None, config_name="cifar10_single_process")
def run(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    trainer_config = cfg.trainer
    model_config = cfg.model
    data_config = cfg.data

    train(
        trainer_config=trainer_config,
        data_config=data_config,
        model_config=model_config,
        use_cuda_if_available=True,
        world_size=1,
    )


if __name__ == "__main__":
    cfg = maybe_parse_json_config()
    run(cfg)
