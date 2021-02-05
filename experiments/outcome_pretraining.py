import os
from pathlib import Path
import random

from farm.eval import Evaluator
from farm.modeling.adaptive_model import AdaptiveModel
from farm.modeling.optimization import initialize_optimizer
from farm.modeling.language_model import LanguageModel
from farm.modeling.prediction_head import TextClassificationHead, NextSentenceHead
from farm.modeling.tokenization import Tokenizer
from farm.train import EarlyStopping
from farm.utils import set_all_seeds, MLFlowLogger, initialize_device_settings
from ray import tune
import yaml
import fire

import utils

from custom_models.training.ExtendedTrainer import ExtendedTrainer
from custom_models.outcome_pretraining.OutcomePretrainingDataSilo import OutcomePretrainingDataSilo
from custom_models.outcome_pretraining.OutcomePretrainingProcessor import OutcomePretrainingProcessor

logger = utils.get_logger(__name__)


def outcome_pretraining(task_config,
                        model_name,
                        cache_dir,
                        run_name="0",
                        lr=1e-05,
                        warmup_steps=5000,
                        embeds_dropout=0.1,
                        epochs=200,  # large because we use early stopping by default
                        batch_size=20,
                        grad_acc_steps=1,
                        early_stopping_metric="loss",
                        early_stopping_mode="min",
                        early_stopping_patience=10,
                        model_class="Bert",
                        tokenizer_class="BertTokenizer",
                        do_lower_case=True,
                        do_train=True,
                        do_eval=True,
                        do_hpo=False,
                        max_seq_len=512,
                        seed=11,
                        eval_every=500,
                        use_amp=False,
                        use_cuda=True,
                        ):
    # Load task config
    task_config = yaml.safe_load(open(task_config))

    data_dir = Path(task_config["data"]["data_dir"])

    # General Settings
    set_all_seeds(seed=seed)
    device, n_gpu = initialize_device_settings(use_cuda=use_cuda, use_amp=use_amp)

    # 1.Create a tokenizer
    tokenizer = Tokenizer.load(pretrained_model_name_or_path=model_name, tokenizer_class=tokenizer_class,
                               do_lower_case=do_lower_case)

    # 2. Create a DataProcessor that handles all the conversion from raw text into a pytorch Dataset
    processor = OutcomePretrainingProcessor(tokenizer=tokenizer,
                                            max_seq_len=max_seq_len,
                                            data_dir=data_dir,
                                            train_filename=task_config["data"]["train_filename"],
                                            dev_filename=task_config["data"]["dev_filename"],
                                            seed=seed,
                                            max_size_admission=50,
                                            max_size_discharge=50,
                                            cache_dir=cache_dir)

    # 3. Create a DataSilo that loads several datasets (train/dev/test), provides DataLoaders for them and calculates a
    #    few descriptive statistics of our datasets
    data_silo = OutcomePretrainingDataSilo(
        processor=processor,
        caching=True,
        cache_dir=cache_dir,
        batch_size=batch_size,
        max_multiprocessing_chunksize=200)

    if do_train:

        # Set save dir for experiment output
        save_dir = Path(task_config["output_dir"]) / f'{task_config["experiment_name"]}_{run_name}'

        # Use HPO config args if config is passed
        if do_hpo:
            save_dir = save_dir / tune.session.get_trial_name()
        else:
            exp_name = f"exp_{random.randint(100000, 999999)}"
            save_dir = save_dir / exp_name

        # Create save dir
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # Setup MLFlow logger
        ml_logger = MLFlowLogger(tracking_uri=task_config["log_dir"])
        ml_logger.init_experiment(experiment_name=task_config["experiment_name"],
                                  run_name=f'{task_config["experiment_name"]}_{run_name}')

        # 4. Create an AdaptiveModel
        # a) which consists of a pretrained language model as a basis

        language_model = LanguageModel.load(model_name, language_model_class=model_class)

        # b) and NextSentenceHead prediction head or TextClassificationHead if it's not a Bert Model
        if model_class == "Bert":
            next_sentence_head = NextSentenceHead.load(model_class)
        else:
            next_sentence_head = TextClassificationHead(num_labels=2)

        model = AdaptiveModel(
            language_model=language_model,
            prediction_heads=[next_sentence_head],
            embeds_dropout_prob=embeds_dropout,
            lm_output_types=["per_sequence"],
            device=device,
        )

        # 5. Create an optimizer
        schedule_opts = {"name": "LinearWarmup",
                         "num_warmup_steps": warmup_steps}

        model, optimizer, lr_schedule = initialize_optimizer(
            model=model,
            learning_rate=lr,
            device=device,
            n_batches=len(data_silo.loaders["train"]),
            n_epochs=epochs,
            use_amp=use_amp,
            grad_acc_steps=grad_acc_steps,
            schedule_opts=schedule_opts)

        # 6. Create an early stopping instance
        early_stopping = None
        if early_stopping_mode != "none":
            early_stopping = EarlyStopping(
                mode=early_stopping_mode,
                min_delta=0.0001,
                save_dir=save_dir,
                metric=early_stopping_metric,
                patience=early_stopping_patience
            )

        # 7. Feed everything to the Trainer, which keeps care of growing our model into powerful plant and evaluates it
        # from time to time

        trainer = ExtendedTrainer(
            model=model,
            optimizer=optimizer,
            data_silo=data_silo,
            epochs=epochs,
            n_gpu=n_gpu,
            lr_schedule=lr_schedule,
            evaluate_every=eval_every,
            early_stopping=early_stopping,
            device=device,
            grad_acc_steps=grad_acc_steps,
            evaluator_test=do_eval
        )

        def score_callback(eval_score, train_loss):
            tune.report(roc_auc_dev=eval_score, train_loss=train_loss)

        # 8. Train the model
        trainer.train(score_callback=score_callback if do_hpo else None)

        # 9. Save model if not saved in early stopping
        model.save(save_dir / "final_model")
        processor.save(save_dir / "final_model")

    if do_eval:
        # Load newly trained model or existing model
        if do_train:
            model_dir = save_dir
        else:
            model_dir = Path(model_name)

        logger.info("###### Eval on TEST SET #####")

        evaluator_test = Evaluator(
            data_loader=data_silo.get_data_loader("test"),
            tasks=data_silo.processor.tasks,
            device=device
        )

        # Load trained model for evaluation
        model = AdaptiveModel.load(model_dir, device)
        model.connect_heads_with_processor(data_silo.processor.tasks, require_labels=True)

        # Evaluate
        results = evaluator_test.eval(model, return_preds_and_labels=True)

        # Log results
        utils.log_results(results, dataset_name="test", steps=len(evaluator_test.data_loader),
                          save_path=model_dir / "eval_results.txt")


if __name__ == '__main__':
    fire.Fire(outcome_pretraining)
