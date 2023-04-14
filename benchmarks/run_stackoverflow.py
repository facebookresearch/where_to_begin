#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Example usages:
FL Training with JSON config
"""
import json
from datetime import timedelta
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

import flsim.configs  # noqa
import hydra
import numpy as np
import torch
import torch.nn as nn
from flsim.data.data_provider import IFLDataProvider, IFLUserData
from flsim.interfaces.batch_metrics import IFLBatchMetrics
from flsim.interfaces.metrics_reporter import Channel
from flsim.metrics_reporter.tensorboard_metrics_reporter import FLMetricsReporter
from flsim.interfaces.model import IFLModel
from flsim.utils.config_utils import maybe_parse_json_config
from flsim.utils.data.data_utils import batchify
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, GPT2Tokenizer
from .utils import set_random_seed


class StackoverflowDataset(Dataset):
    def __init__(self, path, num_users=None):
        with open(path, "+r") as f:
            self.data = json.load(f)
        self.ids = list(self.data.keys())
        if num_users is not None:
            self.ids = self.ids[:num_users]

    def __len__(self):
        return len(self.ids)

    def client_ids(self):
        return self.ids

    def __getitem__(self, index):
        id = self.ids[index]
        return id, self.data[id]


class StackoverflowUserData(IFLUserData):
    def __init__(
        self,
        user_data: Tuple[str, Dict[str, Any]],
        train_batch_size,
        eval_batch_size,
        tokenizer,
        max_seq_len,
        eval_split=0,
        max_samples: Optional[int] = None,
    ):
        assert (
            eval_split == 0 or eval_split == 1.0
        ), "Per user eval split is not supported yet for Stackoverflow"

        self.train_batches = []
        self.eval_batches = []
        self.train_examples = 0
        self.eval_examples = 0
        self.client_id, data = user_data

        if eval_split == 1.0:
            tokens, self.eval_examples = StackoverflowUserData.process_tokens(
                data=data["tokens"],
                tokenizer=tokenizer,
                max_seq_len=max_seq_len,
                max_samples=max_samples,
            )
            eval_batches = list(batchify(tokens, eval_batch_size, drop_last=False))
            self.clean_batches(eval_batches)
            self.eval_batches = eval_batches
        else:
            tokens, self.train_examples = StackoverflowUserData.process_tokens(
                data=data["tokens"],
                tokenizer=tokenizer,
                max_seq_len=max_seq_len,
                max_samples=max_samples,
            )
            train_batches = list(batchify(tokens, train_batch_size, drop_last=False))
            self.clean_batches(train_batches)
            self.train_batches = train_batches

    @staticmethod
    def clean_batches(batches):
        """Remove the last sample from the last batch if it contains
        only one token (in which case there is nothing to predict), where
        NaN value arises as a consequence of computing cross entropy loss
        between two empty tensors.
        """

        if len(batches[-1][-1]) > 1:
            return

        batches[-1].pop()
        if len(batches[-1]) == 0:
            batches.pop()

    @classmethod
    def process_tokens(cls, data, tokenizer, max_seq_len, max_samples):
        tokens = []
        for sent in data:
            if sent != " ":
                # get token index tensor; shape = (num_tokens, )
                sent = tokenizer.tokenize(sent)
                sent = [tokenizer.bos_token] + sent + [tokenizer.eos_token]
                sent = (
                    tokenizer.encode(
                        sent,
                        return_tensors="pt",
                        max_length=max_seq_len,
                        truncation=True,
                    )
                    .flatten()
                    .long()
                )
                tokens.append(sent)

        if max_samples is not None:
            tokens = tokens[:max_samples]
        num_tokens = sum(map(len, tokens))
        return tokens, num_tokens

    def train_data(self) -> Iterator[Dict[str, torch.Tensor]]:
        return self.train_batches

    def eval_data(self):
        return self.eval_batches

    def num_train_batches(self):
        return len(self.train_batches)

    def num_eval_batches(self):
        return len(self.eval_batches)

    def num_train_examples(self) -> int:
        return self.train_examples

    def num_eval_examples(self):
        return self.eval_examples


class StackoverflowDataProvider(IFLDataProvider):
    def __init__(
        self,
        train_dataset: StackoverflowDataset,
        eval_dataset: StackoverflowDataset,
        test_dataset: StackoverflowDataset,
        train_batch_size,
        eval_batch_size,
        tokenizer,
        max_seq_len,
        eval_split=0,
        max_samples: Optional[int] = None,
    ):
        assert (
            eval_split == 0
        ), "Per user eval split is not supported yet for Stackoverflow"
        self._train_users = self._create_fl_users(
            iterator=train_dataset,
            train_batch_size=train_batch_size,
            eval_batch_size=eval_batch_size,
            eval_split=0.0,
            tokenizer=tokenizer,
            max_seq_len=max_seq_len,
            max_samples=max_samples,
        )
        self._eval_users = self._create_fl_users(
            iterator=eval_dataset,
            train_batch_size=train_batch_size,
            eval_batch_size=eval_batch_size,
            eval_split=1.0,
            tokenizer=tokenizer,
            max_seq_len=max_seq_len,
            max_samples=max_samples,
        )
        self._test_users = self._create_fl_users(
            iterator=test_dataset,
            train_batch_size=train_batch_size,
            eval_batch_size=eval_batch_size,
            eval_split=1.0,
            tokenizer=tokenizer,
            max_seq_len=max_seq_len,
            max_samples=max_samples,
        )

    def train_user_ids(self) -> List[int]:
        return list(self._train_users.keys())

    def num_train_users(self) -> int:
        return len(self._train_users)

    def get_train_user(self, user_index: int) -> IFLUserData:
        return self._train_users[user_index]

    def train_users(self) -> Iterable[IFLUserData]:
        return list(self._train_users.values())

    def eval_users(self) -> Iterable[IFLUserData]:
        return list(self._eval_users.values())

    def test_users(self) -> Iterable[IFLUserData]:
        return list(self._test_users.values())

    def _create_fl_users(
        self,
        iterator: Dataset,
        train_batch_size,
        eval_batch_size,
        eval_split,
        tokenizer,
        max_seq_len,
        max_samples,
    ) -> Dict[int, IFLUserData]:
        user_dict = {}
        user_index = 0
        # pyre-fixme[6]: For 1st param expected `Iterable[Variable[_T]]` but got
        #  `Dataset[typing.Any]`.
        for user_data in tqdm(iterator, desc="Creating FL User", unit="user"):
            user = StackoverflowUserData(
                user_data,
                train_batch_size=train_batch_size,
                eval_batch_size=eval_batch_size,
                eval_split=eval_split,
                max_seq_len=max_seq_len,
                max_samples=max_samples,
                tokenizer=tokenizer,
            )
            if user.num_train_batches() > 0 or user.num_eval_batches() > 0:
                user_dict[user_index] = user
                user_index += 1
        return user_dict


class BatchMetrics(IFLBatchMetrics):
    def __init__(
        self,
        *,
        loss: torch.Tensor,
        num_examples: int,
        num_correct: int,
    ) -> None:
        self._loss = loss
        self._num_examples = num_examples
        self.num_correct = num_correct

    @property
    def loss(self) -> torch.Tensor:
        return self._loss

    @property
    def num_examples(self) -> int:
        return self._num_examples

    @property
    def predictions(self) -> torch.Tensor:
        pass

    @property
    def targets(self) -> torch.Tensor:
        pass

    @property
    def model_inputs(self) -> Any:
        pass


class FLModel(IFLModel):
    def __init__(
        self,
        model: nn.Module,
        vocab_size: int,
        device: Optional[str] = None,
    ) -> None:
        # Replace the original embedding module by a new embedding whose vocab size
        # is enlarged by 1 to create room for pad_idx.
        # pyre-fixme[16]: Item `Tensor` of `Union[Tensor, Module]` has no attribute
        #  `wte`.
        cached_embedding = model.transformer.wte.weight[:vocab_size]
        # pyre-fixme[16]: Item `Tensor` of `Union[Tensor, Module]` has no attribute
        #  `wte`.
        dim = model.transformer.wte.weight.shape[1]
        pad_idx = vocab_size
        extended_embedding = nn.Embedding(vocab_size + 1, dim, padding_idx=pad_idx)
        extended_weight = torch.cat([cached_embedding, torch.zeros(1, dim)])
        del cached_embedding
        extended_embedding.load_state_dict({"weight": extended_weight})
        # pyre-fixme[16]: `Module` has no attribute `wte`.
        # pyre-fixme[16]: `Tensor` has no attribute `wte`.
        model.transformer.wte = extended_embedding

        self.model = model
        self.device = device

        self.pad_idx: int = pad_idx
        self.cross_entropy_loss: nn.Module = nn.CrossEntropyLoss(ignore_index=pad_idx)

    def fl_forward(self, batch) -> BatchMetrics:
        tokens = batch
        tokens = torch._C._nn.pad_sequence(
            tokens, batch_first=True, padding_value=float(self.pad_idx)
        )

        if self.device is not None:
            tokens = tokens.to(self.device)

        output, _ = self.model(tokens)
        loss, num_correct, num_examples = self.cross_entropy_eval(output, tokens)
        loss = loss.cpu()
        del output
        del tokens
        return BatchMetrics(
            loss=loss,
            num_examples=num_examples,
            num_correct=num_correct,
        )

    def fl_get_module(self) -> nn.Module:
        return self.model

    def fl_cuda(self) -> None:
        self.model = self.model.to(self.device)  # pyre-ignore

    def get_eval_metrics(self, batch) -> BatchMetrics:
        with torch.no_grad():
            return self.fl_forward(batch)

    def fl_create_training_batch(self, **kwargs):
        pass

    def get_num_examples(self, batch):
        pass

    def cross_entropy_eval(self, lm_logits, labels):
        """
        Routine from Huggingface's GPT-2 implementation (v 4.7.0)
        """
        shift_logits = lm_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        flat_logits = shift_logits.view(-1, shift_logits.size(-1))
        flat_labels = shift_labels.view(-1)
        pred = flat_logits.argmax(1)

        loss = self.cross_entropy_loss(flat_logits, flat_labels)
        valid_mask = flat_labels != self.pad_idx
        num_correct = (pred[valid_mask] == flat_labels[valid_mask]).sum()
        num_examples = valid_mask.sum()

        return (
            loss,
            num_correct,
            num_examples,
        )


class MetricsReporter(FLMetricsReporter):
    PPL = "Perplexity"
    ACCURACY = "Accuracy"

    def __init__(
        self,
        channels: List[Channel],
        log_dir: Optional[str] = None,
    ):
        super().__init__(channels, log_dir)
        self._set_summary_writer(log_dir=log_dir)
        self.num_correct = 0
        self.num_examples = 0
        self.num_batches = 0
        self.losses = []

    def add_batch_metrics(self, metrics: IFLBatchMetrics) -> None:
        self.losses.append(metrics.loss.item())
        self.num_examples += metrics.num_examples
        # pyre-ignore[]
        self.num_correct += metrics.num_correct.item()
        self.num_batches += 1

    def compare_metrics(self, eval_metrics, best_metrics):
        print(f"Current eval accuracy: {eval_metrics}%, Best so far: {best_metrics}%")
        if best_metrics is None:
            return True
        return eval_metrics[self.PPL] < best_metrics[self.PPL]

    def compute_scores(self) -> Dict[str, Any]:
        perplexity = np.exp(np.mean(self.losses))
        accuracy = (self.num_correct / (self.num_examples + 1e-6)) * 100

        if np.isnan(perplexity) or np.isinf(perplexity):
            raise ValueError("Nan loss")
        return {self.PPL: perplexity, self.ACCURACY: accuracy}

    def create_eval_metrics(
        self, scores: Dict[str, Any], total_loss: float, **kwargs
    ) -> Any:
        return scores

    def reset(self):
        self.num_correct = 0
        self.num_examples = 0
        self.num_batches = 0
        self.losses = []


def train(cfg):
    cuda_enabled = torch.cuda.is_available()
    device = torch.device("cuda:0" if cuda_enabled else "cpu")
    set_random_seed(cfg.model.seed, use_cuda=cuda_enabled)

    train_dataset = StackoverflowDataset(
        cfg.data.train_path, cfg.data.num_users, bucket=cfg.data.train_bucket
    )
    eval_dataset = StackoverflowDataset(
        cfg.data.eval_path, cfg.data.num_users, bucket=cfg.data.eval_bucket
    )
    test_dataset = StackoverflowDataset(
        cfg.data.test_path, cfg.data.num_users, bucket=cfg.data.test_bucket
    )
    tokenizer = GPT2Tokenizer.from_pretrained(cfg.data.tokenizer_path)
    data_provider = StackoverflowDataProvider(
        train_dataset,
        eval_dataset,
        test_dataset,
        train_batch_size=cfg.data.train_batch_size,
        eval_batch_size=cfg.data.eval_batch_size,
        tokenizer=tokenizer,
        max_seq_len=cfg.data.max_seq_len,
        max_samples=cfg.data.max_samples,
    )
    if cfg.model.pretrained:
        model = AutoModelForCausalLM.from_pretrained(cfg.model.model_path)
        trainable_params = 0
        for p in model.transformer.parameters():
            p.requires_grad = False
        for p in model.lm_head.parameters():
            p.requires_grad = True
            trainable_params += p.numel()
        print("Note when pretrained is true, we only fine-tune the last layer")
        print(f"Trainable parameters count: {trainable_params / 1e6}M")
    else:
        model = AutoModelForCausalLM.from_config(
            AutoConfig.from_pretrained(cfg.model.model_path)
        )

    # pyre-fixme[6]
    fl_model = FLModel(model, tokenizer.vocab_size, device)
    fl_model.fl_cuda()

    trainer = instantiate(cfg.trainer, model=fl_model)

    metrics_reporter = MetricsReporter([Channel.STDOUT, Channel.TENSORBOARD])

    final_model, eval_metrics = trainer.train(
        data_provider=data_provider,
        metrics_reporter=metrics_reporter,
        num_total_users=data_provider.num_train_users(),
        distributed_world_size=1,
        rank=0,
    )
    test_metrics = trainer.test(
        data_provider=data_provider,
        metrics_reporter=MetricsReporter([Channel.STDOUT]),
    )
    print("Eval ", eval_metrics)
    print("Test ", test_metrics)
    return eval_metrics, test_metrics, metrics_reporter.writer.log_dir


@hydra.main(config_path=None, config_name="stackoverflow")
def run(cfg: DictConfig) -> None:
    train(cfg, rank=0, world_size=1)


if __name__ == "__main__":
    cfg = maybe_parse_json_config()
    run(cfg)
