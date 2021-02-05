import json
import math
import os
import random
import re
from tasks import utils


def get_json_data(path: str):
    with open(path, encoding="utf-8") as sample_file:
        medquad = json.load(sample_file)
    return medquad['documents']


def get_headings():
    with open("admission_headings.txt", encoding="utf-8") as adm_file:
        admission_headings = set(adm_file.read().splitlines())

    with open("discharge_headings.txt", encoding="utf-8") as dis_file:
        discharge_headings = set(dis_file.read().splitlines())

    assert len(admission_headings.intersection(discharge_headings)) == 0

    return admission_headings, discharge_headings


def split_data(medquad: json, seed: int):
    # define split size
    val_split = 0.15
    val_size = math.ceil(len(medquad) * val_split)

    # shuffle
    random.Random(seed).shuffle(medquad)

    # split into train and val
    splits = {
        "train": medquad[val_size:],
        "val": medquad[:val_size]
    }

    print('#train_set:', len(splits['train']), '#val_set:', len(splits['val']))
    return splits


def get_idx_from_disease_symptoms(disease_json, key, value):
    annotator_dict = dict()
    for ann in disease_json['annotations']:
        if key in ann.keys():
            if ann[key] == value:
                annotator_dict[ann['aspect'] + '_' + ann['focus']] = ann['begin'], ann['begin'] + ann['length']
    return annotator_dict


def get_disease_label_text(document, label, idx):
    annotator_dict = get_idx_from_disease_symptoms(document, 'aspect', label)
    if len(annotator_dict):
        excerpt = ''
        for key in annotator_dict:
            start, end = annotator_dict[key]
            text = document['text']
            excerpt = excerpt + text[start:end]

        return excerpt


def split_admission_discharge(medquad_path: str, save_dir: str, seed: int):
    """
    Split texts into admission and discharge sections to prepare them for ADMISSION DISCHARGE MATCHING task.
    """

    # set task name
    task_name = "ADM_DIS_MATCH_MEDQUAD"

    # load dataframes
    medquad = get_json_data(medquad_path)

    # load admission discharge sections
    admission_headings, discharge_headings = get_headings()

    # build splits
    splits = split_data(medquad, seed)

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
                admission_text = get_disease_label_text(sample, heading, idx)

                if admission_text:
                    sample_adm += admission_text + " "
            elif heading in discharge_headings:
                discharge_text = get_disease_label_text(sample, heading, idx)
                if discharge_text:
                    sample_dis += discharge_text + " "

        if sample_adm != "" and sample_dis != "":
            # check if first sentence is a question
            adm_idx = 0
            dis_idx = 0

            # if there is no identified sentence and sent is empty
            try:
                pat = re.compile(r'([A-Z][^\.!?]*[\.!?])', re.M)
                adm_sent = pat.findall(sample_adm)
                adm_idx = adm_sent[0].find('?') + 1
                dis_sent = pat.findall(sample_dis)
                dis_idx = dis_sent[0].find('?') + 1
            except:
                pass
            # remove whitespaces and newlines
            sample_adm = " ".join(sample_adm[adm_idx:].replace('\n', ' ').replace('\r', '').strip().split())
            sample_dis = " ".join(sample_dis[dis_idx:].replace('\n', ' ').replace('\r', '').strip().split())

            if sample_adm != "" and sample_dis != "":
                text += sample_adm
                text += "\n[SEP]\n"
                text += sample_dis
                text += "\n\n"

    text = remove_boilerplate(text)

    return text.strip()


def remove_boilerplate(text):
    BOILERPLATE_1 = "Much of this information comes from Orphanet, a European rare disease database. The frequency of a sign or symptom is usually listed as a rough estimate of the percentage of patients who have that feature. The frequency may also be listed as a fraction. The first number of the fraction is how many people had the symptom, and the second number is the total number of people who were examined in one study. For example, a frequency of 25/25 means that in a study of 25 people all patients were found to have that symptom. Because these frequencies are based on a specific study, the fractions may be different if another group of patients are examined. Sometimes, no information on frequency is available. In these cases, the sign or symptom may be rare or common."

    BOILERPLATE_2 = "If the information is available, the table below includes how often the symptom is seen in people with this condition. You can use the MedlinePlus Medical Dictionary to look up the definitions for these medical terms. Signs and Symptoms Approximate number of patients (when available) "

    BOILERPLATE_3 = "The Human Phenotype Ontology provides the following list of signs and symptoms for "

    BOILERPLATE_4 = "The Human Phenotype Ontology (HPO) has collected information on how often a sign or symptom occurs in a condition"

    cleaned_text = text.replace(BOILERPLATE_1, "").replace(BOILERPLATE_2, "").replace(BOILERPLATE_3, "").replace(
        BOILERPLATE_4, "")

    return cleaned_text


if __name__ == "__main__":
    args = utils.parse_args()
    split_admission_discharge(args.path, # Location of MedQuAD_queries_10_candidates.json
                              args.save_dir,
                              args.seed)
