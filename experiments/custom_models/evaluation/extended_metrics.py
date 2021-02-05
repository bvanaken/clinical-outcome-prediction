import logging
import inspect
import numpy as np
from farm.evaluation.metrics import matthews_corrcoef, simple_accuracy, acc_and_f1, pearson_and_spearman, ner_f1_score, \
    f1_macro, squad, mean_squared_error, r2_score, top_n_accuracy, registered_metrics, text_similarity_metric
from sklearn.metrics import roc_auc_score

logger = logging.getLogger(__name__)


def compute_metrics(metric, preds, probs, labels, multilabel):
    assert len(preds) == len(labels)
    if metric == "mcc":
        return {"mcc": matthews_corrcoef(labels, preds)}
    elif metric == "acc":
        return simple_accuracy(preds, labels)
    elif metric == "acc_f1":
        return acc_and_f1(preds, labels)
    elif metric == "pear_spear":
        return pearson_and_spearman(preds, labels)
    elif metric == "seq_f1":
        return {"seq_f1": ner_f1_score(labels, preds)}
    elif metric == "f1_macro":
        return f1_macro(preds, labels)
    elif metric == "squad":
        return squad(preds, labels)
    elif metric == "mse":
        return {"mse": mean_squared_error(preds, labels)}
    elif metric == "r2":
        return {"r2": r2_score(preds, labels)}
    elif metric == "top_n_accuracy":
        return {"top_n_accuracy": top_n_accuracy(preds, labels)}
    elif metric == "text_similarity_metric":
        return text_similarity_metric(preds, labels)
    elif metric == "roc_auc":
        return {"roc_auc": roc_auc(probs, labels, multilabel=multilabel)}
    elif metric in registered_metrics:
        metric_func = registered_metrics[metric]

        metric_args = inspect.getfullargspec(metric_func).args
        if "probs" and "multilabel" in metric_args:
            return metric_func(preds, probs, labels, multilabel)
        elif "probs" in metric_args:
            return metric_func(preds, probs, labels)
        elif "multilabel" in metric_args:
            return metric_func(preds, labels, multilabel)
        else:
            return metric_func(preds, labels)
    else:
        raise KeyError(metric)


def roc_auc(probs, labels, multilabel=False, average='macro', multi_class='ovo'):
    if isinstance(labels, list):
        labels = np.array(labels, dtype=int)
    else:
        labels = labels.astype(int)

    y_score = probs

    if multilabel:
        # Exclude columns with only one value, e.g. only false
        dim_size = len(labels[0])
        mask = np.ones((dim_size), dtype=bool)
        for c in range(dim_size):
            if max(labels[:, c]) == 0:
                mask[c] = False
        labels = labels[:, mask]
        y_score = np.array(probs)[:, mask]

        filtered_cols = np.count_nonzero(mask == False)
        logger.info(f"{filtered_cols} columns not considered for ROC AUC calculation!")

    return roc_auc_score(y_true=labels, y_score=y_score, average=average, multi_class=multi_class)
