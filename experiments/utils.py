import logging
import os
from pathlib import Path

from farm.utils import MLFlowLogger as MlLogger
import numbers
import random
from farm.visual.ascii.images import BUSH_SEP


def init_save_dir(output_dir, experiment_name, run_name, hpo_trial_name):
    # Set save dir for experiment output
    save_dir = Path(output_dir) / f'{experiment_name}_{run_name}'

    # Use HPO config args if config is passed
    if hpo_trial_name is None:
        exp_name = f"exp_{random.randint(100000, 999999)}"
        save_dir = save_dir / exp_name
    else:
        save_dir = save_dir / hpo_trial_name

    # Create save dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    return save_dir


def save_predictions(results, save_dir, multilabel=False, dataset_name="test"):
    predictions_log = ""
    for head in results:
        prob_per_sample = head["probs"]
        for sample_prob in prob_per_sample:
            if multilabel:
                for label_prob in sample_prob:
                    predictions_log += str(label_prob) + " "

                predictions_log = predictions_log[:-1] + "\n"
            else:
                predictions_log += str(sample_prob) + "\n"

    with open(os.path.join(save_dir, f"predictions_{dataset_name}.txt"), "w") as prediction_file:
        prediction_file.write(predictions_log)


def is_3_digit_code(code):
    if code.startswith("V") or code.startswith("E"):
        code = code[1:]
    if len(code) == 3:
        try:
            int(code)
            return True
        except:
            return False
    return False


def is_i2b2_code(code):
    return code in ["414", "401", "250", "272", "278"]


def get_logger(name):
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO)

    return logging.getLogger(name)


def log_results(results, dataset_name, steps, logging=True, print=True, save_path=None, num_fold=None):
    logger = get_logger(__name__)

    # Print a header
    header = "\n\n"
    header += BUSH_SEP + "\n"
    header += "***************************************************\n"
    if num_fold:
        header += f"***** EVALUATION | FOLD: {num_fold} | {dataset_name.upper()} SET | AFTER {steps} BATCHES *****\n"
    else:
        header += f"***** EVALUATION | {dataset_name.upper()} SET | AFTER {steps} BATCHES *****\n"
    header += "***************************************************\n"
    header += BUSH_SEP + "\n"
    logger.info(header)

    save_log = header

    for head_num, head in enumerate(results):
        logger.info("\n _________ {} _________".format(head['task_name']))
        for metric_name, metric_val in head.items():
            metric_log = None

            # log with ML framework (e.g. Mlflow)
            if logging:
                if not metric_name in ["preds", "probs", "labels"] and not metric_name.startswith("_"):
                    if isinstance(metric_val, numbers.Number):
                        MlLogger.log_metrics(
                            metrics={
                                f"{dataset_name}_{metric_name}_{head['task_name']}": metric_val
                            },
                            step=steps,
                        )

            # print via standard python logger
            if print:
                if metric_name == "report":
                    if isinstance(metric_val, str) and len(metric_val) > 8000:
                        metric_val = metric_val[:7500] + "\n ............................. \n" + metric_val[-500:]
                    metric_log = "{}: \n {}".format(metric_name, metric_val)
                    logger.info(metric_log)
                else:
                    if not metric_name in ["preds", "probs", "labels"] and not metric_name.startswith("_"):
                        metric_log = "{}: {}".format(metric_name, metric_val)
                        logger.info(metric_log)

            if save_path and metric_log:
                save_log += "\n" + metric_log

    if save_path:
        with open(save_path, "w", encoding="utf-8") as log_file:
            log_file.write(save_log)
