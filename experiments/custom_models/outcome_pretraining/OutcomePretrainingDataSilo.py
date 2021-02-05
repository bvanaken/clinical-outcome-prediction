import logging
from pathlib import Path

import torch
from farm.data_handler.data_silo import _StreamingDataSet, DataSilo
from farm.data_handler.dataloader import NamedDataLoader
from farm.visual.ascii.images import TRACTOR_SMALL
from torch.utils.data.sampler import SequentialSampler
import numpy as np
from farm.utils import MLFlowLogger as MlLogger

logger = logging.getLogger(__name__)


class OutcomePretrainingDataSilo(DataSilo):
    """
    Streaming Data Silo loads and preprocesses datasets in parallel to the model training.

    The samples are lazily created from the input file and batches are yielded on-the-fly when required during training.
    This is useful if you:
    - work with large datasets that don't fit in memory
    - want to save time (by not preprocessing the entire dataset before starting training)

    For optimal training performance and efficient utilization of shiny GPUs, the pipeline always keeps a few
    pre-computed batches ready to avoid any waiting time when a batch is requested during training.

    To parallelize the creation of batches, PyTorch DataLoader provide an option to use
    multiple workers that utilize the available CPU cores and ensure enough pre-computed batches.
    """

    def __init__(self, processor, batch_size, distributed=False, automatic_loading=True,
                 max_multiprocessing_chunksize=2000, max_processes=128, caching=False,
                 cache_path=Path("cache/data_silo"), cache_dir=None):
        """
        :param processor: A dataset specific Processor object which will turn input (file or dict) into a Pytorch Dataset.
        :type processor: Processor
        :param batch_size: The size of batch that should be returned by the DataLoaders.
        :type batch_size: int
        :param distributed: Set to True if the program is running in a distributed setting.
        :type distributed: bool
        :param automatic_loading: Set to False, if you don't want to automatically load data at initialization.
        :type automatic_loading: bool
        :param max_multiprocessing_chunksize: max possible value for chunksize as calculated by `calc_chunksize()`
            in `farm.utils`. For certain cases like lm_finetuning, a smaller value can be set, as the default chunksize
            values are rather large that might cause memory issues.
        :type max_multiprocessing_chunksize: int
        :param max_processes: the maximum number of processes to spawn in the multiprocessing.Pool used in DataSilo.
                              It can be set to 1 to disable the use of multiprocessing ot make debugging easier.
        :type max_processes: int
        :param caching: save the processed datasets on disk to save time/compute if the same train data is used to run
                        multiple experiments. Each cache has a checksum based on the train_filename of the Processor
                        and the batch size.
        :type caching: bool
        :param cache_path: root dir for storing the datasets' cache.
        :type cache_path: Path
        """

        self.distributed = distributed
        self.processor = processor
        self.data = {}
        self.batch_size = batch_size
        self.class_weights = None
        self.max_processes = max_processes
        self.max_multiprocessing_chunksize = max_multiprocessing_chunksize
        self.caching = caching
        self.cache_path = cache_path
        self.cache_dir = cache_dir

        if len(self.processor.tasks) == 0:
            raise Exception("No task initialized. Try initializing the processor with a metric and a label list. "
                            "Alternatively you can add a task using Processor.add_task()")

        loaded_from_cache = False
        if self.caching:  # Check if DataSets are present in cache
            if cache_dir:
                dataset_path = self.cache_dir
            else:
                checksum = self._get_checksum()
                dataset_path = self.cache_path / checksum

            test_set_path = dataset_path / "test_dataset"

            if test_set_path.exists():
                self._load_dataset_from_cache(dataset_path)
                loaded_from_cache = True

        if not loaded_from_cache and automatic_loading:
            # In most cases we want to load all data automatically, but in some cases we rather want to do this
            # later or load from dicts instead of file (https://github.com/deepset-ai/FARM/issues/85)
            self._load_data()

    def _load_data(self, train_dicts=None, dev_dicts=None, test_dicts=None):
        """
        Loading the train, dev and test datasets either from files (default) or from supplied dicts.
        The processor is called to handle the full conversion from "raw data" to a Pytorch Dataset.
        The resulting datasets are loaded into DataSilo.data

        :param train_dicts: (Optional) dicts containing examples for training.
        :param dev_dicts: (Optional) dicts containing examples for dev.
        :param test_dicts: (Optional) dicts containing examples for test.
        :return: None
        """
        logger.info("\nLoading data into the data silo ..."
                    "{}".format(TRACTOR_SMALL))

        # dev data
        dev_file = self.processor.data_dir / self.processor.dev_filename
        logger.info("Loading dev set from: {}".format(dev_file))
        self.data["dev"], self.tensor_names = self._get_dataset(dev_file)

        # skip the test data
        self.data["test"] = None

        if self.caching:
            self._save_dataset_to_cache()

        # derive stats and meta data
        self._calculate_statistics()

        self._initialize_data_loaders()

    def _load_dataset_from_cache(self, cache_dir):
        """
        Load serialized dataset from a cache.
        """
        logger.info(f"Loading datasets from cache at {cache_dir}")

        dev_dataset_path = cache_dir / "dev_dataset"
        if dev_dataset_path.exists():
            self.data["dev"] = torch.load(dev_dataset_path)
        else:
            self.data["dev"] = None

        test_dataset_path = cache_dir / "test_dataset"
        if test_dataset_path.exists():
            self.data["test"] = torch.load(test_dataset_path)
        else:
            self.data["test"] = None

        self.tensor_names = torch.load(cache_dir / "tensor_names")

        # derive stats and meta data
        self._calculate_statistics()
        # self.calculate_class_weights()

        self._initialize_data_loaders()

    def _save_dataset_to_cache(self):
        """
        Serialize and save dataset to a cache.
        """
        if self.cache_dir:
            cache_dir = self.cache_dir
        else:
            checksum = self._get_checksum()
            cache_dir = self.cache_path / checksum
        cache_dir.mkdir(parents=True, exist_ok=True)

        torch.save(self.data["dev"], cache_dir / "dev_dataset")

        torch.save(self.data["test"], cache_dir / "test_dataset")

        torch.save(self.tensor_names, cache_dir / "tensor_names")
        logger.info(f"Cached the datasets at {cache_dir}")

    def _initialize_data_loaders(self):
        """ Initializing train, dev and test data loaders for the already loaded datasets """

        if self.data["dev"] is not None:
            data_loader_dev = NamedDataLoader(
                dataset=self.data["dev"],
                sampler=SequentialSampler(self.data["dev"]),
                batch_size=self.batch_size,
                tensor_names=self.tensor_names,
            )
        else:
            data_loader_dev = None

        if self.processor.test_filename:
            data_loader_test = NamedDataLoader(
                dataset=self.data["test"],
                sampler=SequentialSampler(self.data["test"]),
                batch_size=self.batch_size,
                tensor_names=self.tensor_names,
            )
        else:
            data_loader_test = None

        self.loaders = {
            "dev": data_loader_dev,
            "test": data_loader_test,
        }

    def _calculate_statistics(self):
        """ Calculate and log simple summary statistics of the datasets """

        self.counts = {
            "train": 0
        }

        if self.data["dev"]:
            self.counts["dev"] = len(self.data["dev"])
        else:
            self.counts["dev"] = 0

        if self.data["test"]:
            self.counts["test"] = len(self.data["test"])
        else:
            self.counts["test"] = 0

        seq_lens = []
        for dataset in self.data["dev"].datasets:
            train_input_numpy = dataset[:][0].numpy()
            seq_lens.extend(np.sum(train_input_numpy != self.processor.tokenizer.pad_token_id, axis=1))
        max_seq_len = dataset[:][0].shape[1]

        self.clipped = np.mean(np.array(seq_lens) == max_seq_len)
        self.ave_len = np.mean(seq_lens)

        logger.info("Examples in train: {}".format(self.counts["train"]))
        logger.info("Examples in dev  : {}".format(self.counts["dev"]))
        logger.info("Examples in test : {}".format(self.counts["test"]))
        logger.info("")
        logger.info("Longest sequence length observed after clipping:     {}".format(max(seq_lens)))
        logger.info("Average sequence length after clipping: {}".format(self.ave_len))
        logger.info("Proportion clipped:      {}".format(self.clipped))
        if self.clipped > 0.5:
            logger.info("[Farmer's Tip] {}% of your samples got cut down to {} tokens. "
                        "Consider increasing max_seq_len. "
                        "This will lead to higher memory consumption but is likely to "
                        "improve your model performance".format(round(self.clipped * 100, 1), max_seq_len))

        MlLogger.log_params(
            {
                "n_samples_train": self.counts["train"],
                "n_samples_dev": self.counts["dev"],
                "n_samples_test": self.counts["test"],
                "batch_size": self.batch_size,
                "ave_seq_len": self.ave_len,
                "clipped": self.clipped
            }
        )

    def get_data_loader(self, dataset_name):
        """
        Returns a new instance of dataloader for the given dataset.

        The dataloader lazily yields from Iterable DataSets. After a complete iteration
        over the input data, the generators gets exhausted. So, for instance, in the
        case of model training, a new train dataloader must be used for each train epoch.

        :param dataset_name: 'train', 'dev', or 'test' set.
        :type dataset_name: str
        """
        if dataset_name == "train":
            filename = self.processor.train_filename

            #  Batching:
            #
            #  The model Trainer is passed a PyTorch DataLoader instance that yields dataset batches for training.
            #
            #  By default, the PyTorch DataLoader prefetch (2 * num_workers) samples. However, given the higher
            #  batch sizes(usually >64) for model training, the default prefetch is not sufficient to keep the
            #  model Training saturated with datasets.
            #
            #  As a workaround, we yield batches of samples instead of yielding individual samples. The DataLoader
            #  can then prefetch (2 * num_workers) number of batches of samples.
            #
            #  Since the batching is now handled within _StreamingDataSet, we disable the batching on DataLoader side
            #  by initializing the data loader with batch_size as 1.

            data_set = _StreamingDataSet(
                processor=self.processor,
                filepath=self.processor.data_dir / filename,
                batch_size=self.batch_size,
                dataloader_workers=self.max_processes,
            )
            data_loader = NamedDataLoader(
                dataset=data_set, batch_size=1, num_workers=self.max_processes, pin_memory=True
            )
            return data_loader

        else:
            return self.loaders[dataset_name]
