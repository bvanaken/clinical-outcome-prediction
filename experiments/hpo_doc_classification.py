import os

from hyperopt import hp
from ray import tune
from ray.tune.suggest.hyperopt import HyperOptSearch
import yaml
import fire
from doc_classification import doc_classification


def hpo_doc_classification(task_config,
                           model_name,
                           cache_dir,
                           hpo_samples,
                           hpo_gpus,
                           run_name="0",
                           balance_classes=True,
                           check_balance_classes=True,
                           lr_start=5e-6,
                           lr_end=5e-4,
                           warmup_steps_start=30,
                           warmup_steps_end=5000,
                           grad_acc_start=1,
                           grad_acc_end=20,
                           dropout_start=0.1,
                           dropout_end=0.3,
                           metric="roc_auc_dev",
                           mode="max",
                           stopping_criteria_iteration=None,
                           stopping_criteria_value=None,
                           **kwargs):
    # Create a HyperOpt search space
    space = {
        "lr": hp.uniform("lr", lr_start, lr_end),
        "warmup_steps": hp.uniform("warmup_steps", warmup_steps_start, warmup_steps_end),
        "grad_acc": hp.choice("grad_acc", [grad_acc_start, grad_acc_end]),
        "embeds_dropout": hp.uniform("embeds_dropout", dropout_start, dropout_end)
    }

    if check_balance_classes:
        space["balance_classes"] = hp.choice("balance_classes", [True, False])

    # Specify the search space and maximize score
    hyperopt = HyperOptSearch(space, metric=metric, mode=mode)

    # Define stopping criteria
    def stopping_criteria(trial_id, result):
        metric_criteria = result[metric] < stopping_criteria_value if mode == "max" \
            else result[metric] > stopping_criteria_value
        return result["training_iteration"] >= stopping_criteria_iteration and metric_criteria

    # Load task config
    task_config_dict = yaml.safe_load(open(task_config))
    exp_name = f'{task_config_dict["experiment_name"]}_{run_name}'
    log_dir = os.path.join(task_config_dict["output_dir"], exp_name)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    def exp(config):
        doc_classification(task_config=task_config,
                           model_name_or_path=model_name,
                           cache_dir=cache_dir,
                           run_name=run_name,
                           lr=config["lr"],
                           warmup_steps=config["warmup_steps"],
                           balance_classes=config["balance_classes"] if check_balance_classes else balance_classes,
                           embeds_dropout=config["embeds_dropout"],
                           do_hpo=True,
                           **kwargs
                           )

    analysis = tune.run(
        exp,
        stop=stopping_criteria if stopping_criteria_iteration is not None else None,
        search_alg=hyperopt,
        num_samples=hpo_samples,
        resources_per_trial={"gpu": hpo_gpus},
        local_dir=log_dir,
        name=exp_name)

    print("best config: ", analysis.get_best_config(metric="roc_auc_dev", mode="max"))
    print("best trial: ", analysis.get_best_trial(metric="roc_auc_dev", mode="max"))


if __name__ == "__main__":
    fire.Fire(hpo_doc_classification)
