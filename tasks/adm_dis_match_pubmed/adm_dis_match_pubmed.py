import json
import math
import os
import random

from tasks import utils


def split_admission_discharge(pubmed_section_path: str, save_dir: str, seed: int):
    """
    Split texts into admission and discharge sections to prepare them for ADMISSION DISCHARGE MATCHING task.
    """

    # set task name
    task_name = "ADM_DIS_MATCH_PUBMED"

    # load dataframes
    with open(pubmed_section_path, encoding="utf-8") as sample_file:
        pubmed_data = json.load(sample_file)

    # load admission discharge sections
    with open("admission_headings.txt", encoding="utf-8") as adm_file:
        admission_headings = adm_file.read().splitlines()

    with open("discharge_headings.txt", encoding="utf-8") as dis_file:
        discharge_headings = dis_file.read().splitlines()

    # get papers following PICO schema that contain one of the admission sections
    pico_papers = get_pico_papers(pubmed_data, admission_headings)

    # build splits

    # define split size
    val_split = 0.15
    val_size = math.ceil(len(pico_papers) * val_split)

    # shuffle
    random.Random(seed).shuffle(pico_papers)

    # split into train and val
    splits = {
        "train": pico_papers[val_size:],
        "val": pico_papers[:val_size]
    }

    for split_name in splits:
        text = build_text_from_samples(splits[split_name], admission_headings, discharge_headings)

        file_name = f"{os.path.join(save_dir, task_name)}_{split_name}.txt"
        with open(file_name, "w", encoding="utf-8") as write_file:
            write_file.write(text)


def clean_heading(heading):
    heading = heading.lower()
    return ''.join([i for i in heading if i.isalpha() or i == " "]).strip()


def get_section_text(entry, annotation):
    text = entry["text"][annotation["begin"]:annotation["begin"] + annotation["length"]]
    return text


def get_pico_papers(data, admission_sections):
    papers = []

    for entry in data:
        anns = entry["annotations"]
        for ann in anns:
            heading = clean_heading(ann["sectionHeading"])
            if heading in admission_sections:
                papers.append(entry)
                break

    return papers


def build_text_from_samples(samples, admission_sections, discharge_sections):
    text = ""
    for entry in samples:
        sample_adm = ""
        sample_dis = ""
        anns = entry["annotations"]
        for ann in anns:
            heading = clean_heading(ann["sectionHeading"])
            if heading in admission_sections or \
                    "features" in heading or \
                    ("diagnosis" in heading and "differential" not in heading):
                sample_adm += get_section_text(entry, ann) + " "
                continue
            if heading in discharge_sections or \
                    "treatment" in heading or \
                    "findings" in heading or \
                    "complications" in heading or \
                    "procedure" in heading or \
                    "management" in heading or \
                    "followup" in heading or \
                    "therapy" in heading or \
                    "differential" in heading:
                sample_dis += get_section_text(entry, ann) + " "

        sample_adm = sample_adm.strip()
        sample_dis = sample_dis.strip()

        if sample_adm != "" and sample_dis != "":
            adm_dis_text = sample_adm.replace("[", "") + "\n[SEP]\n" + sample_dis.replace("[", "")
            text += adm_dis_text + "\n\n"

    return text.strip()


if __name__ == "__main__":
    args = utils.parse_args()
    split_admission_discharge(args.path,
                              args.save_dir,
                              args.seed)
