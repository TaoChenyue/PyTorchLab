from typing import Any, Mapping

import torch
import yaml
from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import Callback
from torch import Tensor
from torchmetrics.classification import PrecisionRecallCurve
from torchvision.utils import make_grid, save_image

from pytorchlab.callbacks.utils import get_batch_save_path
from pytorchlab.typehints import OutputsDict


class AnomalyCallback(Callback):
    def __init__(
        self,
        threshold: float | None = None,
        pixel_threshold: float = 0.5,
        score_name: str = "anomaly_score",
        label_name: str = "label",
        image_name: str = "reconstructed",
        image_nums: int = 1,
        **kwargs,
    ):
        """
        _summary_
        threshold,min_score,max_score should be define in LightningModule
        save_hyperparams() should be used
        OutputsDict:
            {
                inputs:{
                    images:{
                      <image_name>: tensor
                    },
                    labels:{
                        <label_name>: tensor
                    }
                },
                outputs:{
                    images:{
                        <image_name>: tensor
                    }
                }
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
        self.threshold_suggest = threshold
        self.pixel_threshold = pixel_threshold
        self.min_score = None
        self.max_score = None
        self.score_name = score_name
        self.label_name = label_name
        self.image_name = image_name
        self.image_nums = image_nums
        self.reset_metrics()
        self.kwargs = kwargs

    def reset_metrics(self):
        self.score_list = []
        self.label_list = []
        self.pr_curve.reset()

    def on_save_checkpoint(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        checkpoint: torch.Dict[str, Any],
    ) -> None:
        checkpoint.update(
            {
                "threshold": self.threshold_suggest,
                "min_score": self.min_score,
                "max_score": self.max_score,
            }
        )
        return super().on_save_checkpoint(trainer, pl_module, checkpoint)

    def _get_score(self, outputs: OutputsDict):
        return outputs.get("metrics", {}).get(self.score_name, None)

    def _get_label(self, outputs: OutputsDict):
        return outputs.get("inputs", {}).get("labels", {}).get(self.label_name, None)

    def _get_pr_threshold(self):
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
        if self.threshold is None:
            index = torch.argmax(f1_score)
        else:
            index = torch.min(torch.nonzero(f1_score >= self.threshold))
        self.threshold_suggest = threshold[torch.argmax(f1_score)]
        return (
            pred[index],
            recall[index],
        )

    def _load_paras(self, trainer: Trainer, pl_module: LightningModule):
        if trainer.ckpt_path is not None:
            checkpoint = torch.load(trainer.ckpt_path, map_location=pl_module.device)
            self.threshold_suggest = checkpoint.get("threshold", 0.5)
            self.min_score = checkpoint.get("min_score", 0)
            self.max_score = checkpoint.get("max_score", 1)

    def _get_heatmap(self, outputs: OutputsDict):
        image = outputs.get("inputs", {}).get("images", {}).get(self.image_name, None)
        reconstructed = (
            outputs.get("outputs", {}).get("images", {}).get(self.image_name, None)
        )
        if image is None or reconstructed is None:
            return None
        heatmaps = torch.abs(image - reconstructed)
        heatmaps[heatmaps < self.pixel_threshold] = 0
        heatmaps[heatmaps > self.pixel_threshold] = 1
        return heatmaps

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
        self.score_list.append(score)
        self.label_list.append(label)

    def on_validation_epoch_end(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        if len(self.score_list) == 0:
            return

        pred, recall = self._get_pr_threshold()
        pl_module.log_dict(
            {
                "precision": pred,
                "recall": recall,
            },
            sync_dist=True,
        )

    def on_test_epoch_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self.reset_metrics()
        self._load_paras(trainer, pl_module)

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
            return
        self.score_list.append(score)
        self.label_list.append(label)

    def on_test_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if len(self.score_list) == 0:
            return
        pred, recall = self._get_pr_threshold()
        pl_module.log_dict(
            {
                "precision": pred,
                "recall": recall,
            },
            sync_dist=True,
        )

    def on_predict_epoch_start(
        self, trainer: Trainer, pl_module: LightningModule
    ) -> None:
        self._load_paras(trainer, pl_module)

    def on_predict_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: OutputsDict,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        score = self._get_score(outputs)
        score = (score - self.min_score) / (self.max_score - self.min_score)
        threshold = self.threshold if self.threshold else self.threshold_suggest
        heatmap = self._get_heatmap(outputs)
        save_path = get_batch_save_path(trainer, pl_module, batch_idx, dataloader_idx)
        save_image(
            make_grid(heatmap[: self.image_nums], **self.kwargs),
            save_path / "heatmap.png",
        )
        input_label = self._get_label(outputs)
        input_label = [
            "abnormal" if label == 1 else "normal"
            for label in input_label[: self.image_nums]
        ]
        yaml.dump(input_label, open(save_path / "input_label.yaml", "w"))
        label = [
            "abnormal" if score >= threshold else "normal"
            for score in score[: self.image_nums]
        ]
        yaml.dump(label, open(save_path / "output_label.yaml", "w"))
