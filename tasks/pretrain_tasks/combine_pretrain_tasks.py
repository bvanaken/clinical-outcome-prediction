import glob
import os
import random
from typing import List

from tasks import utils


def split_into_samples(text):
    return text.split("\n\n")


def read_split_file(data_dir, split):
    path = glob.glob(os.path.join(data_dir, f"*_{split}.txt"))
    with open(path[0], encoding="utf-8") as read_file:
        return read_file.read()


def write_file(text, path):
    with open(path, "w", encoding="utf-8") as write_file:
        write_file.write(text)


def build_combined_pretraining_set(task_name: str, data_dirs: List, save_dir: str, seed: int):
    name = f"PRETRAIN_{task_name}"

    train_samples = []
    val_samples = []

    # collect samples
    for data_dir in data_dirs:
        train_file = read_split_file(data_dir, "train")
        val_file = read_split_file(data_dir, "val")

        train_samples += split_into_samples(train_file)
        val_samples += split_into_samples(val_file)

    # shuffle
    random.Random(seed).shuffle(train_samples)
    random.Random(seed).shuffle(val_samples)

    # save
    train_text = "\n\n".join(train_samples)
    write_file(train_text, os.path.join(save_dir, f"{name}_train.txt"))

    val_text = "\n\n".join(val_samples)
    write_file(val_text, os.path.join(save_dir, f"{name}_val.txt"))


if __name__ == "__main__":
    args = utils.parse_args_combine()
    build_combined_pretraining_set(args.task_name, args.data_dirs, args.save_dir, args.seed)
