import logging
import sys
import torch
from tqdm import tqdm
from farm.visual.ascii.images import GROWING_TREE
from farm.train import Trainer

from custom_models.evaluation.ExtendedEvaluator import ExtendedEvaluator

from .ExtendedAdaptiveModel import ExtendedAdaptiveModel

logger = logging.getLogger(__name__)


class ExtendedTrainer(Trainer):
    def train(self, score_callback=None):
        """
        Perform the training procedure.

        The training is visualized by a progress bar. It counts the epochs in a zero based manner.
        For example, when you specify ``epochs=20`` it starts to count from 0 to 19.

        If trainer evaluates the model with a test set the result of the
        evaluation is stored in ``test_result``.

        :return: Returns the model after training. When you do ``early_stopping``
            with a ``save_dir`` the best model is loaded and returned.
        """

        # connect the prediction heads with the right output from processor
        self.model.connect_heads_with_processor(self.data_silo.processor.tasks, require_labels=True)
        # Check that the tokenizer fits the language model
        # TODO: make this compliant for DP / DDP where the model class is wrapped
        if self.model._get_name() == 'BiAdaptiveModel':
            self.model.verify_vocab_size(vocab_size1=len(self.data_silo.processor.tokenizer),
                                         vocab_size2=len(self.data_silo.processor.passage_tokenizer))
        else:
            self.model.verify_vocab_size(vocab_size=len(self.data_silo.processor.tokenizer))
        self.model.train()

        do_stopping = False
        evalnr = 0
        loss = 0
        resume_from_step = self.from_step

        if self.local_rank in [0, -1]:
            logger.info(f"\n {GROWING_TREE}")

        for epoch in range(self.from_epoch, self.epochs):
            early_break = False
            self.from_epoch = epoch
            train_data_loader = self.data_silo.get_data_loader("train")
            progress_bar = tqdm(train_data_loader, disable=self.local_rank not in [0, -1] or self.disable_tqdm)
            for step, batch in enumerate(progress_bar):
                # when resuming training from a checkpoint, we want to fast forward to the step of the checkpoint
                if resume_from_step and step <= resume_from_step:
                    # TODO: Improve skipping for StreamingDataSilo
                    # The seeds before and within the loop are currently needed, if you need full reproducibility
                    # of runs with vs. without checkpointing using StreamingDataSilo. Reason: While skipping steps in StreamingDataSilo,
                    # we update the state of the random number generator (e.g. due to masking words), which can impact the model behaviour (e.g. dropout)
                    if step % 10000 == 0:
                        logger.info(f"Skipping {step} out of {resume_from_step} steps ...")
                    if resume_from_step == step:
                        logger.info(f"Finished skipping {resume_from_step} steps ...")
                        resume_from_step = None
                    else:
                        continue

                progress_bar.set_description(f"Train epoch {epoch}/{self.epochs - 1} (Cur. train loss: {loss:.4f})")

                # Only for distributed training: we need to ensure that all ranks still have a batch left for training
                if self.local_rank != -1:
                    if not self._all_ranks_have_data(has_data=1, step=step):
                        early_break = True
                        break

                # Move batch of samples to device
                batch = {key: batch[key].to(self.device) for key in batch}

                # Forward & backward pass through model
                logits = self.model.forward(**batch)
                per_sample_loss = self.model.logits_to_loss(logits=logits, global_step=self.global_step, **batch)
                loss = self.backward_propagate(per_sample_loss, step)

                # Perform  evaluation
                if self.evaluate_every != 0 \
                        and self.global_step % self.evaluate_every == 0 \
                        and self.global_step != 0 \
                        and self.local_rank in [0, -1]:
                    # When using StreamingDataSilo, each evaluation creates a new instance of
                    # dev_data_loader. In cases like training from scratch, this could cause
                    # some variance across evaluators due to the randomness in word masking.
                    dev_data_loader = self.data_silo.get_data_loader("dev")
                    if dev_data_loader is not None:
                        evaluator_dev = ExtendedEvaluator(
                            data_loader=dev_data_loader, tasks=self.data_silo.processor.tasks, device=self.device,
                            report=self.eval_report
                        )
                        evalnr += 1
                        result = evaluator_dev.eval(self.model)
                        evaluator_dev.log_results(result, "Dev", self.global_step)
                        if self.early_stopping:
                            do_stopping, save_model, eval_value = self.early_stopping.check_stopping(result)

                            # Call score callback if assigned
                            if score_callback:
                                score_callback(eval_score=eval_value, train_loss=loss)

                            if save_model:
                                logger.info(
                                    "Saving current best model to {}, eval={}".format(
                                        self.early_stopping.save_dir, eval_value))
                                self.model.save(self.early_stopping.save_dir)
                                self.data_silo.processor.save(self.early_stopping.save_dir)
                            if do_stopping:
                                # log the stopping
                                logger.info(
                                    "STOPPING EARLY AT EPOCH {}, STEP {}, EVALUATION {}".format(epoch, step, evalnr))
                if do_stopping:
                    break

                self.global_step += 1
                self.from_step = step + 1

                # save the current state as a checkpoint before exiting if a SIGTERM signal is received
                if self.sigterm_handler and self.sigterm_handler.kill_now:
                    logger.info("Received a SIGTERM signal. Saving the current train state as a checkpoint ...")
                    if self.local_rank in [0, -1]:
                        self._save()
                        torch.distributed.destroy_process_group()
                        sys.exit(0)

                # save a checkpoint and continue train
                if self.checkpoint_every and step % self.checkpoint_every == 0:
                    if self.local_rank in [0, -1]:
                        self._save()
                    # Let other ranks wait until rank 0 has finished saving
                    if self.local_rank != -1:
                        torch.distributed.barrier()

            if do_stopping:
                break

            # Only for distributed training: we need to ensure that all ranks still have a batch left for training
            if self.local_rank != -1 and not early_break:
                self._all_ranks_have_data(has_data=False)

        # With early stopping we want to restore the best model
        if self.early_stopping and self.early_stopping.save_dir:
            logger.info("Restoring best model so far from {}".format(self.early_stopping.save_dir))
            lm_name = self.model.language_model.name
            self.model = ExtendedAdaptiveModel.load(self.early_stopping.save_dir, self.device, lm_name=lm_name)
            self.model.connect_heads_with_processor(self.data_silo.processor.tasks, require_labels=True)

        # Eval on test set
        if self.evaluator_test:
            test_data_loader = self.data_silo.get_data_loader("test")
            if test_data_loader is not None:
                evaluator_test = ExtendedEvaluator(
                    data_loader=test_data_loader, tasks=self.data_silo.processor.tasks, device=self.device
                )
                self.test_result = evaluator_test.eval(self.model)
                evaluator_test.log_results(self.test_result, "Test", self.global_step)
        return self.model
