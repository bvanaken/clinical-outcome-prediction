from farm.eval import Evaluator
from tqdm import tqdm
import torch
import logging
import numpy as np
from farm.evaluation.metrics import compute_report_metrics
from farm.utils import to_numpy
from farm.modeling.adaptive_model import AdaptiveModel

from .extended_metrics import compute_metrics

logger = logging.getLogger(__name__)


class ExtendedEvaluator(Evaluator):
    def eval(self, model, return_preds_and_labels=False):
        """
        Performs evaluation on a given model.
        :param model: The model on which to perform evaluation
        :type model: AdaptiveModel
        :param return_preds_and_labels: Whether to add preds and labels in the returned dicts of the
        :type return_preds_and_labels: bool
        :return all_results: A list of dictionaries, one for each prediction head. Each dictionary contains the metrics
                             and reports generated during evaluation.
        :rtype all_results: list of dicts
        """
        model.eval()

        # init empty lists per prediction head
        loss_all = [0 for _ in model.prediction_heads]
        preds_all = [[] for _ in model.prediction_heads]
        probs_all = [[] for _ in model.prediction_heads]
        label_all = [[] for _ in model.prediction_heads]
        ids_all = [[] for _ in model.prediction_heads]
        passage_start_t_all = [[] for _ in model.prediction_heads]

        for step, batch in enumerate(
                tqdm(self.data_loader, desc="Evaluating", mininterval=10)
        ):
            batch = {key: batch[key].to(self.device) for key in batch}

            with torch.no_grad():

                logits = model.forward(**batch)
                losses_per_head = model.logits_to_loss_per_head(logits=logits, **batch)
                preds = model.logits_to_preds(logits=logits, **batch)
                probs = model.logits_to_probs(logits=logits, **batch)
                labels = model.prepare_labels(**batch)

            # stack results of all batches per prediction head
            for head_num, head in enumerate(model.prediction_heads):
                loss_all[head_num] += np.sum(to_numpy(losses_per_head[head_num]))
                preds_all[head_num] += list(to_numpy(preds[head_num]))
                probs_all[head_num] += list(to_numpy(probs[head_num]))
                label_all[head_num] += list(to_numpy(labels[head_num]))
                if head.model_type == "span_classification":
                    ids_all[head_num] += list(to_numpy(batch["id"]))
                    passage_start_t_all[head_num] += list(to_numpy(batch["passage_start_t"]))

        # Evaluate per prediction head
        all_results = []
        for head_num, head in enumerate(model.prediction_heads):
            if head.model_type == "multilabel_text_classification":
                # converting from string preds back to multi-hot encoding
                from sklearn.preprocessing import MultiLabelBinarizer
                mlb = MultiLabelBinarizer(classes=head.label_list)
                # TODO check why .fit() should be called on predictions, rather than on labels
                preds_all[head_num] = mlb.fit_transform(preds_all[head_num])
                label_all[head_num] = mlb.transform(label_all[head_num])
            if hasattr(head, 'aggregate_preds'):
                # Needed to convert NQ ids from np arrays to strings
                ids_all_str = [x.astype(str) for x in ids_all[head_num]]
                ids_all_list = [list(x) for x in ids_all_str]
                head_ids = ["-".join(x) for x in ids_all_list]
                preds_all[head_num], label_all[head_num] = head.aggregate_preds(preds=preds_all[head_num],
                                                                                labels=label_all[head_num],
                                                                                passage_start_t=passage_start_t_all[
                                                                                    head_num],
                                                                                ids=head_ids)

            result = {"loss": loss_all[head_num] / len(self.data_loader.dataset),
                      "task_name": head.task_name}
            result.update(
                compute_metrics(metric=head.metric, preds=preds_all[head_num],
                                probs=probs_all[head_num],
                                labels=label_all[head_num],
                                multilabel=head.model_type == "multilabel_text_classification"
                                )
            )

            # Select type of report depending on prediction head output type
            if self.report:
                try:
                    result["report"] = compute_report_metrics(head, preds_all[head_num], label_all[head_num])
                except:
                    logger.error(f"Couldn't create eval report for head {head_num} with following preds and labels:"
                                 f"\n Preds: {preds_all[head_num]} \n Labels: {label_all[head_num]}")
                    result["report"] = "Error"

            if return_preds_and_labels:
                result["preds"] = preds_all[head_num]
                result["probs"] = probs_all[head_num]
                result["labels"] = label_all[head_num]

            all_results.append(result)

        return all_results
