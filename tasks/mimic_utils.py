import argparse

import pandas as pd
import os
import csv


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mimic_dir', required=True)
    parser.add_argument('--save_dir', required=True)
    parser.add_argument('--admission_only', default=False)
    parser.add_argument('--seed', default=123, type=int)

    return parser.parse_args()


def filter_notes(notes_df: pd.DataFrame, admissions_df: pd.DataFrame, admission_text_only=False) -> pd.DataFrame:
    """
    Keep only Discharge Summaries and filter out Newborn admissions. Replace duplicates and join reports with
    their addendums. If admission_text_only is True, filter all sections that are not known at admission time.
    """
    # filter out newborns
    adm_grownups = admissions_df[admissions_df.ADMISSION_TYPE != "NEWBORN"]
    notes_df = notes_df[notes_df.HADM_ID.isin(adm_grownups.HADM_ID)]

    # remove notes with no TEXT or HADM_ID
    notes_df = notes_df.dropna(subset=["TEXT", "HADM_ID"])

    # filter discharge summaries
    notes_df = notes_df[notes_df.CATEGORY == "Discharge summary"]

    # remove duplicates and keep the later ones
    notes_df = notes_df.sort_values(by=["CHARTDATE"])
    notes_df = notes_df.drop_duplicates(subset=["TEXT"], keep="last")

    # combine text of same admissions (those are usually addendums)
    combined_adm_texts = notes_df.groupby('HADM_ID')['TEXT'].apply(lambda x: '\n\n'.join(x)).reset_index()
    notes_df = notes_df[notes_df.DESCRIPTION == "Report"]
    notes_df = notes_df[["HADM_ID", "ROW_ID", "SUBJECT_ID", "CHARTDATE"]]
    notes_df = notes_df.drop_duplicates(subset=["HADM_ID"], keep="last")
    notes_df = pd.merge(combined_adm_texts, notes_df, on="HADM_ID", how="inner")

    # strip texts from leading and trailing and white spaces
    notes_df["TEXT"] = notes_df["TEXT"].str.strip()

    # remove entries without admission id, subject id or text
    notes_df = notes_df.dropna(subset=["HADM_ID", "SUBJECT_ID", "TEXT"])

    if admission_text_only:
        # reduce text to admission-only text
        notes_df = filter_admission_text(notes_df)

    return notes_df


def filter_admission_text(notes_df) -> pd.DataFrame:
    """
    Filter text information by section and only keep sections that are known on admission time.
    """
    admission_sections = {
        "CHIEF_COMPLAINT": "chief complaint:",
        "PRESENT_ILLNESS": "present illness:",
        "MEDICAL_HISTORY": "medical history:",
        "MEDICATION_ADM": "medications on admission:",
        "ALLERGIES": "allergies:",
        "PHYSICAL_EXAM": "physical exam:",
        "FAMILY_HISTORY": "family history:",
        "SOCIAL_HISTORY": "social history:"
    }

    # replace linebreak indicators
    notes_df['TEXT'] = notes_df['TEXT'].str.replace(r"\n", r"\\n")

    # extract each section by regex
    for key in admission_sections.keys():
        section = admission_sections[key]
        notes_df[key] = notes_df.TEXT.str.extract(r'(?i){}(.+?)\\n\\n[^(\\|\d|\.)]+?:'
                                                  .format(section))

        notes_df[key] = notes_df[key].str.replace(r'\\n', r' ')
        notes_df[key] = notes_df[key].str.strip()
        notes_df[key] = notes_df[key].fillna("")
        notes_df[notes_df[key].str.startswith("[]")][key] = ""

    # filter notes with missing main information
    notes_df = notes_df[(notes_df.CHIEF_COMPLAINT != "") | (notes_df.PRESENT_ILLNESS != "") |
                        (notes_df.MEDICAL_HISTORY != "")]

    # add section headers and combine into TEXT_ADMISSION
    notes_df = notes_df.assign(TEXT="CHIEF COMPLAINT: " + notes_df.CHIEF_COMPLAINT.astype(str)
                                    + '\n\n' +
                                    "PRESENT ILLNESS: " + notes_df.PRESENT_ILLNESS.astype(str)
                                    + '\n\n' +
                                    "MEDICAL HISTORY: " + notes_df.MEDICAL_HISTORY.astype(str)
                                    + '\n\n' +
                                    "MEDICATION ON ADMISSION: " + notes_df.MEDICATION_ADM.astype(str)
                                    + '\n\n' +
                                    "ALLERGIES: " + notes_df.ALLERGIES.astype(str)
                                    + '\n\n' +
                                    "PHYSICAL EXAM: " + notes_df.PHYSICAL_EXAM.astype(str)
                                    + '\n\n' +
                                    "FAMILY HISTORY: " + notes_df.FAMILY_HISTORY.astype(str)
                                    + '\n\n' +
                                    "SOCIAL HISTORY: " + notes_df.SOCIAL_HISTORY.astype(str))

    return notes_df


def save_mimic_split_patient_wise(df, label_column, save_dir, task_name, seed, column_list=None):
    """
    Splits a MIMIC dataframe into 70/10/20 train, val, test with no patient occuring in more than one set.
    Uses ROW_ID as ID column and save to save_path.
    """
    if column_list is None:
        column_list = ["ID", "TEXT", label_column]

    # Load prebuilt MIMIC patient splits
    data_split = {"train": pd.read_csv("tasks/mimic_train.csv"),
                  "val": pd.read_csv("tasks/mimic_val.csv"),
                  "test": pd.read_csv("tasks/mimic_test.csv")}

    # Use row id as general id and cast to int
    df = df.rename(columns={'HADM_ID': 'ID'})
    df.ID = df.ID.astype(int)

    # Create path to task data
    os.makedirs(save_dir, exist_ok=True)

    # Save splits to data folder
    for split_name in ["train", "val", "test"]:
        split_set = df[df.SUBJECT_ID.isin(data_split[split_name].SUBJECT_ID)].sample(frac=1,
                                                                                     random_state=seed)[column_list]

        # lower case column names
        split_set.columns = map(str.lower, split_set.columns)

        split_set.to_csv(os.path.join(save_dir, "{}_{}.csv".format(task_name, split_name)),
                         index=False,
                         quoting=csv.QUOTE_ALL)
