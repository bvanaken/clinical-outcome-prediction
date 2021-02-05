from typing import Any

from farm.modeling.adaptive_model import AdaptiveModel
from farm.utils import MLFlowLogger as MlLogger


class ExtendedAdaptiveModel(AdaptiveModel):
    def logits_to_loss(self, logits, global_step=None, **kwargs):
        """
        Get losses from all prediction heads & reduce to single loss *per sample*.

        :param logits: logits, can vary in shape and type, depending on task
        :type logits: object
        :param global_step: number of current training step
        :type global_step: int
        :param kwargs: placeholder for passing generic parameters.
                       Note: Contains the batch (as dict of tensors), when called from Trainer.train().
        :type kwargs: object
        :return loss: torch.tensor that is the per sample loss (len: batch_size)
        """
        all_losses = self.logits_to_loss_per_head(logits, **kwargs)
        # This aggregates the loss per sample across multiple prediction heads
        # Default is sum(), but you can configure any fn that takes [Tensor, Tensor ...] and returns [Tensor]

        # Log the loss per task
        for i, per_sample_loss in enumerate(all_losses):
            task_name = self.prediction_heads[i].task_name
            task_loss = per_sample_loss.mean()
            MlLogger.log_metrics(
                {f"train_loss_{task_name}": float(task_loss.detach().cpu().numpy())},
                step=global_step
            )

        loss = self.loss_aggregation_fn(all_losses, global_step=global_step, batch=kwargs)
        return loss

    def logits_to_probs(self, logits, **kwargs):
        """
        Get probabilities from all prediction heads.
        :param logits: logits, can vary in shape and type, depending on task
        :type logits: object
        :param label_maps: Maps from label encoding to label string
        :param label_maps: dict
        :return: A list of all probabilities from all prediction heads
        """
        all_probs = []
        # collect preds from all heads
        for head, logits_for_head in zip(self.prediction_heads, logits):
            probs = head.logits_to_probs(logits=logits_for_head,
                                         return_class_probs=head.num_labels > 2,
                                         **kwargs)
            all_probs.append(probs)
        return all_probs
