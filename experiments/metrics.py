from sklearn.metrics import matthews_corrcoef, f1_score
from farm.evaluation.metrics import simple_accuracy, register_metrics

import utils

from custom_models.evaluation.extended_metrics import roc_auc

logger = utils.get_logger(__name__)


def binary_classification_metrics(preds, probs, labels, multilabel):
    acc = simple_accuracy(preds, labels).get("acc")
    roc_auc_score = roc_auc(probs=probs, labels=labels, multilabel=multilabel)
    roc_auc_score_weighted = roc_auc(probs=probs, labels=labels, average="weighted", multilabel=multilabel)
    f1macro = f1_score(y_true=labels, y_pred=preds, average="macro")
    f1micro = f1_score(y_true=labels, y_pred=preds, average="micro")

    if not multilabel:
        f1_0 = f1_score(y_true=labels, y_pred=preds, pos_label="0")
        f1_1 = f1_score(y_true=labels, y_pred=preds, pos_label="1")
        mcc = matthews_corrcoef(labels, preds)
    else:
        f1_0, f1_1, mcc = None, None, None
    return {
        "acc": acc,
        "roc_auc": roc_auc_score,
        "roc_auc_weighted": roc_auc_score_weighted,
        "f1_macro": f1macro,
        "f1_micro": f1micro,
        "f1_0": f1_0,
        "f1_1": f1_1,
        "mcc": mcc
    }


def multiclass_classification_metrics(preds, probs, labels):
    acc = simple_accuracy(preds, labels).get("acc")
    roc_auc_score = roc_auc(probs=probs, labels=labels, multi_class='ovo')
    roc_auc_score_weighted = roc_auc(probs=probs, labels=labels, average='weighted', multi_class='ovo')
    f1macro = f1_score(y_true=labels, y_pred=preds, average="macro")
    f1micro = f1_score(y_true=labels, y_pred=preds, average="micro")
    mcc = matthews_corrcoef(labels, preds)

    return {
        "acc": acc,
        "roc_auc": roc_auc_score,
        "roc_auc_weighted": roc_auc_score_weighted,
        "f1_macro": f1macro,
        "f1_micro": f1micro,
        "mcc": mcc
    }


def register_task_metrics(label_list):
    register_metrics('binary_classification_metrics', binary_classification_metrics)
    register_metrics('multiclass_classification_metrics', multiclass_classification_metrics)

    register_multilabel_classification_metrics_3_digits_only('binary_classification_metrics_3_digits_only', label_list)
    register_multilabel_classification_metrics_i2b2_only('binary_classification_metrics_i2b2_only', label_list)


def register_multilabel_classification_metrics_3_digits_only(metric_name, label_list):
    def multilabel_classification_metrics_3_digits_only(preds, probs, labels, multilabel):
        mask = list(map(utils.is_3_digit_code, label_list))
        logger.info(f"Evaluate on {mask.count(True)} 3-digit-codes.")

        return binary_classification_metrics(preds[:, mask], [prob[mask] for prob in probs], labels[:, mask],
                                             multilabel)

    register_metrics(metric_name, multilabel_classification_metrics_3_digits_only)


def register_multilabel_classification_metrics_i2b2_only(metric_name, label_list):
    def multilabel_classification_metrics_i2b2_only(preds, probs, labels, multilabel):
        mask = list(map(utils.is_i2b2_code, label_list))
        logger.info(f"Evaluate on {mask.count(True)} i2b2 codes.")

        return binary_classification_metrics(preds[:, mask], [prob[mask] for prob in probs], labels[:, mask],
                                             multilabel)

    register_metrics(metric_name, multilabel_classification_metrics_i2b2_only)
