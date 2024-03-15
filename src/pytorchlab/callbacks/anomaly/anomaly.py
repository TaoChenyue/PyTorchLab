from typing import Any, Mapping

import torch
from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback
from torch import Tensor
from torchmetrics.classification import PrecisionRecallCurve

from pytorchlab.typehints import OutputsDict


class AnomalyCallback(Callback):
    def __init__(
        self,
        threshold: float | None = None,
        score_name: str = "anomaly_score",
        label_name: str = "label",
    ):
        """
        _summary_
        OutputsDict:
            {
                inputs:{
                    labels:{
                        <label_name>: tensor
                    }
                },
                metrics:{
                    <score_name>: tensor
                },
            }

        Args:
            threshold (float | None, optional): _description_. Defaults to None.
            score_name (str, optional): _description_. Defaults to "anomaly_score".
            label_name (str, optional): _description_. Defaults to "label".
        """
        super().__init__()
        self.pr_curve = PrecisionRecallCurve(task="binary")
        self.threshold = threshold
        self.score_name = score_name
        self.label_name = label_name
        self.reset_metrics()

    def reset_metrics(self):
        self.min_score = float("inf")
        self.max_score = float("-inf")
        self.score_list = []
        self.label_list = []

    def _get_score(self, outputs: OutputsDict):
        return outputs.get("metrics", {}).get(self.score_name, None)

    def _get_label(self, outputs: OutputsDict):
        return outputs.get("inputs", {}).get("labels", {}).get(self.label_name, None)

    def _normalize(self, score: float) -> float:
        return (score - self.min_score) / (self.max_score - self.min_score)

    def on_validation_epoch_start(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        self.reset_metrics()

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
            return
        self.min_score = min(self.min_score, torch.max(score))
        self.max_score = max(self.max_score, torch.max(score))
        self.score_list.append(score)
        self.label_list.append(label)

    def on_validation_epoch_end(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        if len(self.score_list) == 0:
            return
        self.score_list = [self._normalize(x) for x in self.score_list]
        for score, label in zip(self.score_list, self.label_list):
            self.pr_curve.update(score, label)
        pred, recall, threshold = self.pr_curve.compute()
        f1_score = 2 * (pred * recall) / (pred + recall)
        threshold = threshold[torch.argmax(f1_score)]
        max_precision = pred[torch.argmax(f1_score)]
        max_recall = recall[torch.argmax(f1_score)]
        pl_module.hparams.update(
            {
                "threshold": threshold,
                "min_score": self.min_score,
                "max_score": self.max_score,
            }
        )
        pl_module.log_dict(
            {
                "threshold": threshold,
                "min_score": self.min_score,
                "max_score": self.max_score,
                "precision": max_precision,
                "recall": max_recall,
            },
            sync_dist=True,
        )

    # def on_test_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
    #     self.reset_metrics()

    # def on_test_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
    #     self.score_list = [self._normalize(x) for x in self.score_list]
    #     for score, label in zip(self.score_list, self.label_list):
    #         print(score, label)
    #         self.pr_curve.update(score, label)
    #     pred, recall, threshold = self.pr_curve.compute()
    #     f1_score = 2 * (pred * recall) / (pred + recall)
    #     threshold = threshold[torch.argmax(f1_score)]
    #     max_precision = pred[torch.argmax(f1_score)]
    #     max_recall = recall[torch.argmax(f1_score)]
    #     pl_module.log_dict(
    #         {
    #             "threshold": threshold,
    #             "min_score": self.min_score,
    #             "max_score": self.max_score,
    #             "precision": max_precision,
    #             "recall": max_recall,
    #         },
    #         sync_dist=True,
    #     )

    # def on_test_batch_end(
    #     self,
    #     trainer: Trainer,
    #     pl_module: LightningModule,
    #     outputs: Tensor | Mapping[str, Any] | None,
    #     batch: Any,
    #     batch_idx: int,
    #     dataloader_idx: int = 0,
    # ) -> None:
    #     score = self._get_score(outputs)
    #     label = self._get_label(outputs)
    #     if score is None or label is None:
    #         return
    #     self.min_score = min(self.min_score, torch.max(score))
    #     self.max_score = max(self.max_score, torch.max(score))
    #     self.score_list.append(score)
    #     self.label_list.append(label)
