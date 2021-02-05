import json
import math
import os
import random

from tasks import utils


def split_admission_discharge(mt_samples_path: str, save_dir: str, seed: int):
    """
    Split texts into admission and discharge sections to prepare them for ADMISSION DISCHARGE MATCHING task.
    """

    # set task name
    task_name = "ADM_DIS_MATCH_MTSAMPLES"

    # load dataframes
    with open(mt_samples_path, encoding="utf-8") as sample_file:
        mt_samples = json.load(sample_file)

    # load admission discharge sections
    with open("admission_headings.txt", encoding="utf-8") as adm_file:
        admission_headings = adm_file.read().splitlines()

    with open("discharge_headings.txt", encoding="utf-8") as dis_file:
        discharge_headings = dis_file.read().splitlines()

    # build splits

    # define split size
    val_split = 0.15
    val_size = math.ceil(len(mt_samples) * val_split)

    # shuffle
    random.Random(seed).shuffle(mt_samples)

    # split into train and val
    splits = {
        "train": mt_samples[val_size:],
        "val": mt_samples[:val_size]
    }

    for split_name in splits:
        text = build_text_from_samples(splits[split_name], admission_headings, discharge_headings)

        file_name = f"{os.path.join(save_dir, task_name)}_{split_name}.txt"
        with open(file_name, "w", encoding="utf-8") as write_file:
            write_file.write(text)


def build_text_from_samples(samples, admission_headings, discharge_headings):
    text = ""
    for sample in samples:
        sample_adm = ""
        sample_dis = ""
        sections = sample["sections"]

        for section in sections:
            heading = section["heading"]

            # add to sample admission or discharge text
            if heading in admission_headings:
                sample_adm += heading + ": " + section["content"] + " "
            elif heading in discharge_headings:
                sample_dis += heading + ": " + section["content"] + " "

        if sample_adm != "" and sample_dis != "":
            text += sample_adm.replace("\n", "").strip()
            text += "\n[SEP]\n"
            text += sample_dis.replace("\n", "").strip()
            text += "\n\n"

    return text.strip()


if __name__ == "__main__":
    args = utils.parse_args()
    split_admission_discharge(args.path,
                              args.save_dir,
                              args.seed)
