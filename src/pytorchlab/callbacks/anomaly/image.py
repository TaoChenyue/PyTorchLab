from typing import Any, Mapping

import torch
import yaml
from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback
from torch import Tensor
from torchmetrics.classification import PrecisionRecallCurve

from pytorchlab.callbacks.utils import get_batch_save_path, get_epoch_save_path
from pytorchlab.typehints import OutputsDict


class ImageAnomalyCallback(Callback):
    def __init__(
        self,
        score_name: str = "score",
        label_name: str = "label",
        min_score: float | None = None,
        max_score: float | None = None,
        score_threshold: float | None = None,
        **kwargs,
    ):
        super().__init__()
        self.score_name = score_name
        self.label_name = label_name
        self.pr_curve = PrecisionRecallCurve(task="binary")
        self.min_score = min_score
        self.max_score = max_score
        self.score_threshold = score_threshold
        self.suggest_dict = {}
        self.kwargs = kwargs

    def _get_score(self, outputs: OutputsDict):
        return outputs.get("outputs", {}).get("metrics", {}).get(self.score_name, None)

    def _get_label(self, outputs: OutputsDict):
        return outputs.get("inputs", {}).get("labels", {}).get(self.label_name, None)

    def on_save_checkpoint(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        checkpoint: torch.Dict[str, Any],
    ) -> None:
        checkpoint.update(self.suggest_dict)

    def on_validation_epoch_start(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        self.score_list = []
        self.label_list = []
        self.pr_curve.reset()

    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Tensor | Mapping[str, Any] | None,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        score = self._get_score(outputs)
        label = self._get_label(outputs)
        if score is None or label is None:
            raise ValueError("Score and label must not be None.")
        self.score_list.append(score)
        self.label_list.append(label)

    def on_validation_epoch_end(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        self.max_score = max([torch.max(x) for x in self.score_list])
        self.min_score = min([torch.min(x) for x in self.score_list])

        self.score_list = [
            (score - self.min_score) / (self.max_score - self.min_score)
            for score in self.score_list
        ]
        for score, label in zip(self.score_list, self.label_list):
            self.pr_curve.update(score, label)

        pred, recall, threshold = self.pr_curve.compute()
        f1_score = 2 * pred * recall / (pred + recall)

        self.suggest_dict = {
            "max_score": float(self.max_score),
            "min_score": float(self.min_score),
            "score_threshold": float(threshold[torch.argmax(f1_score)]),
        }
        save_path = get_epoch_save_path(trainer, pl_module)
        yaml.dump(self.suggest_dict, open(save_path / "suggest.yaml", "w"))

        if self.score_threshold is None:
            score_threshold = self.suggest_dict["score_threshold"]

        index = torch.min(torch.nonzero(threshold >= score_threshold))
        pl_module.log_dict(
            {
                "precision": pred[index],
                "recall": recall[index],
            },
            sync_dist=True,
        )

    def on_test_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self.score_list = []
        self.label_list = []
        self.pr_curve.reset()

    def on_test_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: OutputsDict,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        score = self._get_score(outputs)
        label = self._get_label(outputs)
        if score is None or label is None:
            raise ValueError("Score and label must not be None.")
        self.score_list.append(score)
        self.label_list.append(label)

    def on_test_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self.max_score = max([torch.max(x) for x in self.score_list])
        self.min_score = min([torch.min(x) for x in self.score_list])

        self.score_list = [
            (score - self.min_score) / (self.max_score - self.min_score)
            for score in self.score_list
        ]
        for score, label in zip(self.score_list, self.label_list):
            self.pr_curve.update(score, label)

        pred, recall, threshold = self.pr_curve.compute()
        f1_score = 2 * pred * recall / (pred + recall)

        self.suggest_dict = {
            "max_score": float(self.max_score),
            "min_score": float(self.min_score),
            "score_threshold": float(threshold[torch.argmax(f1_score)]),
        }
        save_path = get_epoch_save_path(trainer, pl_module)
        yaml.dump(self.suggest_dict, open(save_path / "suggest.yaml", "w"))

        if self.score_threshold is None:
            score_threshold = self.suggest_dict["score_threshold"]

        index = torch.min(torch.nonzero(threshold >= score_threshold))
        pl_module.log_dict(
            {
                "precision": pred[index],
                "recall": recall[index],
            },
            sync_dist=True,
        )

    def on_predict_epoch_start(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        if trainer.ckpt_path is not None:
            ckpt = torch.load(trainer.ckpt_path, map_location=pl_module.device)
            if self.max_score is None:
                self.max_score = ckpt.get("max_score", None)
            if self.min_score is None:
                self.min_score = ckpt.get("min_score", None)
            if self.score_threshold is None:
                self.score_threshold = ckpt.get("score_threshold", None)

        if self.max_score is None:
            self.max_score = self.suggest_dict.get("max_score", None)
        if self.min_score is None:
            self.min_score = self.suggest_dict.get("min_score", None)
        if self.score_threshold is None:
            self.score_threshold = self.suggest_dict.get("score_threshold", None)

        if (
            self.max_score is None
            or self.min_score is None
            or self.score_threshold is None
        ):
            raise ValueError("max_score, min_score and score_threshold must be set")

    def on_predict_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: OutputsDict,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        save_path = get_batch_save_path(trainer, pl_module, batch_idx, dataloader_idx)
        if save_path is None:
            return

        score = self._get_score(outputs)
        score = (score - self.min_score) / (self.max_score - self.min_score)
        pred = score >= self.score_threshold
        output_label = [bool(i) for i in pred]
        input_label = [bool(i) for i in self._get_label(outputs)]
        yaml.dump(input_label, open(save_path / "input_label.yaml", "w"))
        yaml.dump(output_label, open(save_path / "output_label.yaml", "w"))
