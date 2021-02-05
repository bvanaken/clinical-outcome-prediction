import json
import math
import os
import random
import re
from tasks import utils


def get_wiki_data(wiki_path):
    with open(wiki_path, encoding="utf-8") as sample_file:
        wiki = json.load(sample_file)
    return wiki


def get_headings():
    with open("admission_headings.txt", encoding="utf-8") as adm_file:
        admission_headings = set(adm_file.read().splitlines())

    with open("discharge_headings.txt", encoding="utf-8") as dis_file:
        discharge_headings = set(dis_file.read().splitlines())

    assert len(admission_headings.intersection(discharge_headings)) == 0

    return admission_headings, discharge_headings


def get_idx_from_disease_symptoms(disease_json, key, value):
    annotator_dict = dict()
    for ann in disease_json['annotations']:
        if ann[key] == value:
            annotator_dict[ann['sectionLabel'] + '_' + ann['sectionHeading']] = ann['begin'], ann['begin'] + ann[
                'length']
    return annotator_dict


def get_disease_label_text(document, label):
    annotator_dict = get_idx_from_disease_symptoms(document, 'sectionLabel', label)
    if len(annotator_dict):
        excerpt = ''
        for key in annotator_dict:
            start, end = annotator_dict[key]
            text = document['text']
            excerpt = excerpt + text[start:end]

        return excerpt


def split_data(wiki_data: list, seed: int):
    # define split size
    val_split = 0.15
    val_size = math.ceil(len(wiki_data) * val_split)

    # shuffle
    random.Random(seed).shuffle(wiki_data)

    # split into train and val
    splits = {
        "train": wiki_data[val_size:],
        "val": wiki_data[:val_size]
    }

    print('#train_set:', len(splits['train']), '#val_set:', len(splits['val']))
    return splits


def split_admission_discharge(wiki_dir_path: str, save_dir: str, seed: int):
    """
    Split texts into admission and discharge sections to prepare them for ADMISSION DISCHARGE MATCHING task.
    """

    # set task name
    task_name = "ADM_DIS_MATCH_WIKI"

    # load dataframes
    wiki_data = list()
    for sub_path in ['train', 'validation', 'test']:
        wiki_data.extend(get_wiki_data(os.path.join(wiki_dir_path, f'wikisection_en_disease_{sub_path}.json')))

    # load admission discharge sections
    admission_headings, discharge_headings = get_headings()

    # build splits
    splits = split_data(wiki_data, seed)

    for split_name in splits:
        text = build_text_from_samples(splits[split_name],
                                       admission_headings,
                                       discharge_headings)

        file_name = f"{os.path.join(save_dir, task_name)}_{split_name}.txt"

        with open(file_name, "w", encoding="utf-8") as write_file:
            write_file.write(text)


def build_text_from_samples(samples, admission_headings, discharge_headings):
    text = ""
    for idx, sample in enumerate(samples):
        sample_adm = ""
        sample_dis = ""

        headings = admission_headings.union(discharge_headings)

        for heading in headings:

            # add to sample admission or discharge text
            if heading in admission_headings:
                admission_text = get_disease_label_text(sample, heading)
                if admission_text:
                    sample_adm += admission_text + " "

            elif heading in discharge_headings:
                discharge_text = get_disease_label_text(sample, heading)
                if discharge_text:
                    sample_dis += discharge_text + " "

        if sample_adm != "" and sample_dis != "":
            # remove whitespaces and newlines
            sample_adm = " ".join(sample_adm.replace('\n', ' ').replace('\r', '').strip().split())
            sample_dis = " ".join(sample_dis.replace('\n', ' ').replace('\r', '').strip().split())
            text += sample_adm
            text += "\n[SEP]\n"
            text += sample_dis
            text += "\n\n"

    return text.strip()


if __name__ == "__main__":
    args = utils.parse_args()
    split_admission_discharge(args.path,
                              args.save_dir,
                              args.seed)