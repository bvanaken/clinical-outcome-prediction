import logging
import math
import random
from pathlib import Path

import torch
from farm.data_handler.processor import Processor
from farm.data_handler.samples import Sample, SampleBasket
from farm.data_handler.utils import pad
from farm.modeling.tokenization import tokenize_with_metadata, truncate_sequences
from tqdm import tqdm
import numpy as np

logger = logging.getLogger(__name__)


class OutcomePretrainingProcessor(Processor):
    """
    Prepares data for matching sentence prediction in the style of BERT
    """

    def __init__(
            self,
            tokenizer,
            max_seq_len,
            data_dir,
            train_filename="train.txt",
            dev_filename="dev.txt",
            test_filename=None,
            dev_split=0.0,
            max_docs=None,
            seed=123,
            max_size_admission=384,
            max_size_discharge=128,
            cache_dir=None,
            task_name="nextsentence",
            **kwargs
    ):
        """
        :param tokenizer: Used to split a sentence (str) into tokens.
        :param max_seq_len: Samples are truncated after this many tokens.
        :type max_seq_len: int
        :param data_dir: The directory in which the train and dev files can be found. Squad has a private test file
        :type data_dir: str
        :param label_list: list of labels to predict (strings). For most cases this should be: ["start_token", "end_token"]
        :type label_list: list
        :param metric: name of metric that shall be used for evaluation, e.g. "acc" or "f1_macro".
                 Alternatively you can also supply a custom function, that takes preds and labels as args and returns a numerical value.
                 For using multiple metrics supply them as a list, e.g ["acc", my_custom_metric_fn].
        :type metric: str, function, or list
        :param train_filename: The name of the file containing training data.
        :type train_filename: str
        :param dev_filename: The name of the file containing the dev data. If None and 0.0 < dev_split < 1.0 the dev set
                             will be a slice of the train set.
        :type dev_filename: str or None
        :param test_filename: None
        :type test_filename: str or None
        :param dev_split: The proportion of the train set that will sliced. Only works if dev_filename is set to None
        :type dev_split: float
        :param max_docs: maximum number of documents to include from input dataset
        :type max_docs: int
        :param kwargs: placeholder for passing generic parameters
        :type kwargs: object
        """

        self.delimiter = ""
        self.max_docs = max_docs
        self.seed = seed
        self.max_size_admission = max_size_admission
        self.max_size_discharge = max_size_discharge
        self.cache_dir = cache_dir

        super(OutcomePretrainingProcessor, self).__init__(
            tokenizer=tokenizer,
            max_seq_len=max_seq_len,
            train_filename=train_filename,
            dev_filename=dev_filename,
            test_filename=test_filename,
            dev_split=dev_split,
            data_dir=data_dir,
            tasks={}
        )

        self.add_task(task_name, metric="acc", label_list=["False", "True"], label_name="text_classification",
                      label_column_name="nextsentence_label_ids")

    def file_to_dicts(self, file: Path):

        if self.cache_dir:
            file_cache_path = self.cache_dir / f"dicts_{file.stem}"

            if file_cache_path.exists():
                logger.info("Loading dicts from cache.")
                return torch.load(file_cache_path)

        logger.info("Start converting file to dicts.")
        dicts = read_admission_discharge_from_txt(filename=file, tokenizer=self.tokenizer,
                                                  max_size_admission=self.max_size_admission,
                                                  max_size_discharge=self.max_size_discharge)
        # Save dicts
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            dicts = list(dicts)
            torch.save(dicts, file_cache_path)

        return dicts

    def parts_to_sample(self, admission_part, discharge_part, label) -> Sample:
        tokenized = {"text_a": admission_part, "text_b": discharge_part}
        sample_in_clear_text = {
            "text_a": admission_part["clear_text"],
            "text_b": discharge_part["clear_text"],
            "nextsentence_label": label,
        }

        # truncate to max_seq_len
        for seq_name in ["tokens", "offsets", "start_of_word"]:
            tokenized["text_a"][seq_name], tokenized["text_b"][seq_name], _ = truncate_sequences(
                seq_a=tokenized["text_a"][seq_name],
                seq_b=tokenized["text_b"][seq_name],
                tokenizer=self.tokenizer,
                max_seq_len=self.max_seq_len)

        return Sample(id=None, clear_text=sample_in_clear_text, tokenized=tokenized)

    def _dict_to_samples(self, dictionary, all_dicts=None):

        admission = dictionary["admission"]
        discharge = dictionary["discharge"]

        samples = []

        # create samples for each admission part in the doc
        for i_adm in range(len(admission)):
            admission_part = admission[i_adm]

            for i_dis in range(len(discharge)):
                # for each match create one non-matching sample from a random discharge part
                discharge_part = discharge[i_dis]

                matching_sample = self.parts_to_sample(admission_part, discharge_part, label=True)
                samples.append(matching_sample)

                if len(all_dicts) > 1:
                    random_discharge_part = _get_random_discharge_part(all_dicts, forbidden_doc=dictionary)
                    non_matching_sample = self.parts_to_sample(admission_part, random_discharge_part, label=False)
                    samples.append(non_matching_sample)

        return samples

    def _dict_to_samples_fewer_samples(self, dictionary, all_dicts=None):

        admission = dictionary["admission"]
        discharge = dictionary["discharge"]

        random.shuffle(discharge)

        samples = []

        # create samples for each admission part in the doc
        for i_adm in range(len(admission)):
            admission_part = admission[i_adm]

            if i_adm < len(discharge):
                i_dis = i_adm
            else:
                i_dis = random.randint(0, len(discharge) - 1)

            # for each match create one non-matching sample from a random discharge part
            discharge_part = discharge[i_dis]

            matching_sample = self.parts_to_sample(admission_part, discharge_part, label=True)
            samples.append(matching_sample)

            if len(all_dicts) > 1:
                random_discharge_part = _get_random_discharge_part(all_dicts, forbidden_doc=dictionary)
                non_matching_sample = self.parts_to_sample(admission_part, random_discharge_part, label=False)
                samples.append(non_matching_sample)

        return samples

    def _sample_to_features(self, sample) -> list:
        features = samples_to_features_admission_discharge_match(
            sample=sample, max_seq_len=self.max_seq_len, tokenizer=self.tokenizer)
        return features

    def _init_samples_in_baskets(self, fewer_samples=True):
        for basket in self.baskets:
            all_dicts = [b.raw for b in self.baskets]
            try:
                if fewer_samples:
                    basket.samples = self._dict_to_samples_fewer_samples(dictionary=basket.raw, all_dicts=all_dicts)
                else:
                    basket.samples = self._dict_to_samples(dictionary=basket.raw, all_dicts=all_dicts)

                random.shuffle(basket.samples)

                for num, sample in enumerate(basket.samples):
                    sample.id = f"{basket.id_internal}-{num}"
            except Exception as e:
                logger.error(f"Could not create sample(s) from this dict: \n {basket.raw}")
                logger.error(f"{e}")
        baskets_to_remove = [b.id_internal for b in self.baskets if len(b.samples) == 0]
        if baskets_to_remove:
            logger.warning(
                f"Baskets with the following ids have been removed because they have no Samples: {baskets_to_remove}")
        self.baskets = [b for b in self.baskets if len(b.samples) > 0]

    def dataset_from_dicts(self, dicts, indices=None, rest_api_schema=False, return_baskets=False, fewer_samples=True):
        """
        Contains all the functionality to turn a list of dict objects into a PyTorch Dataset and a
        list of tensor names. This can be used for inference mode.

        :param dicts: List of dictionaries where each contains the data of one input sample.
        :type dicts: list of dicts
        :return: a Pytorch dataset and a list of tensor names.
        """
        if rest_api_schema:
            id_prefix = "infer"
        else:
            id_prefix = "train"
        # We need to add the index (coming from multiprocessing chunks) to have a unique basket ID
        if indices:
            self.baskets = [
                SampleBasket(raw=tr, id_internal=f"{id_prefix}-{index}")
                for (tr, index) in zip(dicts, indices)
            ]
        else:
            self.baskets = [
                SampleBasket(raw=tr, id_internal=f"{id_prefix}-{i}")
                for (i, tr) in enumerate(dicts)
            ]
        self._init_samples_in_baskets(fewer_samples=fewer_samples)
        self._featurize_samples()

        if indices:
            logger.info(f"Currently working on indices: {indices}")

            if 0 in indices:
                self._log_samples(2)
            if 50 in indices:
                self._print_samples(30)
        else:
            self._log_samples(2)
        if return_baskets:
            dataset, tensor_names = self._create_dataset(keep_baskets=True)
            return dataset, tensor_names, self.baskets
        else:
            dataset, tensor_names = self._create_dataset()
            return dataset, tensor_names

    def _log_samples(self, n_samples):
        logger.info("*** Show {} random examples ***".format(n_samples))
        for i in range(n_samples):
            random_basket = random.choice(self.baskets)
            while random_basket.samples is None:
                random_basket = random.choice(self.baskets)
            random_sample = random.choice(random_basket.samples)
            logger.info(random_sample)

    def _print_samples(self, n_samples):
        logger.info("*** Print {} random examples ***".format(n_samples))
        log_text = ""
        for i in range(n_samples):
            random_basket = random.choice(self.baskets)
            random_sample = random.choice(random_basket.samples)
            log_text += str(random_sample.clear_text) + "\n\n"

        # with open(os.path.join(self.data_dir, "samples.txt"), "w") as log_file:
        #     log_file.write(log_text)

    def estimate_n_samples(self, filepath, max_docs=500):
        """
        Estimates the number of samples from a given file BEFORE preprocessing.
        Used in StreamingDataSilo to estimate the number of steps before actually processing the data.
        The estimated number of steps will impact some types of Learning Rate Schedules.
        :param filepath: str or Path, file with data used to create samples (e.g. train.txt)
        :param max_docs: int, maximum number of docs to read in & use for our estimate of n_samples
        :return: int, number of samples in the given dataset
        """

        total_lines = sum(1 for line in open(filepath, encoding="utf-8"))
        empty_lines = sum(1 if line == "\n" else 0 for line in open(filepath, encoding="utf-8"))

        n_samples = total_lines - (2 * empty_lines)

        return n_samples


def _get_random_discharge_part(all_baskets, forbidden_doc):
    """
    Get random discharge part from another document for admission_discharge match task.

    :return: str, content of one line
    """
    # Similar to original BERT tf repo: This outer loop should rarely go for more than one iteration for large
    # corpora. However, just to be careful, we try to make sure that
    # the random document is not the same as the document we're processing.
    sentence = None
    for _ in range(100):
        rand_doc_idx = random.randrange(len(all_baskets))
        rand_doc = all_baskets[rand_doc_idx]["discharge"]

        # check if our picked random doc is really different to our initial doc
        if rand_doc != forbidden_doc:
            rand_sent_idx = random.randrange(len(rand_doc))
            sentence = rand_doc[rand_sent_idx]
            break
    if sentence is None:
        raise Exception("Failed to pick out a suitable random substitute for next sentence")
    return sentence


def samples_to_features_admission_discharge_match(sample, max_seq_len, tokenizer):
    """
    Convert a raw sample (pair of sentences as tokenized strings) into a proper training sample with
    IDs, LM labels, padding_mask, CLS and SEP tokens etc.

    :param sample: Sample, containing sentence input as strings and is_next label
    :type sample: Sample
    :param max_seq_len: Maximum length of sequence.
    :type max_seq_len: int
    :param tokenizer: Tokenizer
    :return: InputFeatures, containing all inputs and labels of one sample as IDs (as used for model training)
    """

    tokens_a = sample.tokenized["text_a"]["tokens"]
    tokens_b = sample.tokenized["text_b"]["tokens"]

    # Convert is_next_label: Note that in Bert, is_next_labelid = 0 is used for next_sentence=true!
    if sample.clear_text["nextsentence_label"]:
        is_next_label_id = [0]
    else:
        is_next_label_id = [1]

    # encode string tokens to input_ids and add special tokens
    inputs = tokenizer.encode_plus(text=tokens_a,
                                   text_pair=tokens_b,
                                   add_special_tokens=True,
                                   max_length=max_seq_len,
                                   truncation_strategy='do_not_truncate',
                                   # We've already truncated our tokens before
                                   return_special_tokens_mask=True
                                   )

    input_ids, special_tokens_mask = inputs["input_ids"], inputs["special_tokens_mask"]

    # Use existing segment ids or set them according to a and b text length (with special tokens considered)
    segment_ids = inputs["token_type_ids"] if "token_type_ids" in inputs else [0] * (len(tokens_a) + 2) + [1] * (
            len(tokens_b) + 2)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    padding_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    # Padding up to the sequence length.
    # Normal case: adding multiple 0 to the right
    # Special cases:
    # a) xlnet pads on the left and uses  "4" for padding token_type_ids
    if tokenizer.__class__.__name__ == "XLNetTokenizer":
        pad_on_left = True
        segment_ids = pad(segment_ids, max_seq_len, 4, pad_on_left=pad_on_left)
    else:
        pad_on_left = False
        segment_ids = pad(segment_ids, max_seq_len, 0, pad_on_left=pad_on_left)

    input_ids = pad(input_ids, max_seq_len, tokenizer.pad_token_id, pad_on_left=pad_on_left)
    padding_mask = pad(padding_mask, max_seq_len, 0, pad_on_left=pad_on_left)

    sample_id = random.randint(0, 1000000)

    feature_dict = {"input_ids": input_ids,
                    "padding_mask": padding_mask,
                    "segment_ids": segment_ids,
                    "text_classification_ids": is_next_label_id,
                    "sample_id": sample_id}

    assert len(input_ids) == max_seq_len
    assert len(padding_mask) == max_seq_len
    assert len(segment_ids) == max_seq_len

    return [feature_dict]


def read_admission_discharge_from_txt(filename, tokenizer, doc_delimiter="", in_doc_delimiter="[SEP]",
                                      max_size_admission=384, max_size_discharge=128, encoding="utf-8"):
    """Reads a text file with documents separated by one empty line. Each document has an admission part and a
    discharge part separated by the in_doc_delimiter.
    The admission part is split into even chunks with max size of max_size_discharge and the discharge part respectively
    with max_size_discharge."""

    doc_count = 0
    admission = []
    discharge = []
    is_admission = True
    corpus_lines = 0

    with open(filename, "r", encoding=encoding) as f:
        for line_num, line in enumerate(tqdm(f, desc="Loading Dataset", total=corpus_lines)):
            line = line.strip()
            if line == doc_delimiter:
                if len(admission) > 0 and len(discharge) > 0:
                    yield {"admission": admission, "discharge": discharge}

                    doc_count += 1
                    is_admission = True
                    admission = []
                    discharge = []
                else:
                    logger.warning(f"Found empty document in file (line {line_num}). "
                                   f"Make sure that you comply with the format: "
                                   f"One sentence per line and exactly *one* empty line between docs. "
                                   f"You might have multiple subsequent empty lines.")
            elif line == in_doc_delimiter:
                is_admission = False
            else:
                if is_admission:
                    admission = split_text_token_wise_with_metadata(line, tokenizer, max_chunk_size=max_size_admission)
                else:
                    discharge = split_text_token_wise_with_metadata(line, tokenizer, max_chunk_size=max_size_discharge)

        if doc_count < 2:
            raise ValueError(f"Found only {doc_count} docs in {filename}). You need at least 2! \n"
                             f"Make sure that you comply with the format: \n"
                             f"-> One sentence per line and exactly *one* empty line between docs. \n"
                             f"You might have a single block of text without empty lines inbetween.")


def split_text_token_wise_with_metadata(text, tokenizer, min_chunk_size=30, max_chunk_size=100):
    tokenized_text = tokenize_with_metadata(text, tokenizer)

    token_len = len(tokenized_text["tokens"])
    chunk_size = random.randint(min_chunk_size, max_chunk_size)

    # calculate nr of even chunks with chunksize < chunk_size
    nr_of_chunks = math.ceil(token_len / chunk_size)

    chunks = []

    for i, key in enumerate(tokenized_text.keys()):
        key_chunks = np.array_split(np.array(tokenized_text[key]), nr_of_chunks)

        # update each dict with chunked key
        for j in range(nr_of_chunks):
            if len(chunks) > j:
                chunks[j][key] = key_chunks[j].tolist()
            else:
                chunks.append({key: key_chunks[j].tolist()})

    # reconstruct clear text from offsets
    for k in range(nr_of_chunks):
        chunks[k]["clear_text"] = text[chunks[k]["offsets"][0]: chunks[k]["offsets"][-1] + 1]

    return chunks
