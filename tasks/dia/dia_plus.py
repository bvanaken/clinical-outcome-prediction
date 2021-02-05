from typing import List
import sys

sys.path.append('.')
from tasks import mimic_utils
import pandas as pd
import os

import string
from nltk.corpus import stopwords
from functools import reduce


def dia_plus_mimic(mimic_dir: str, save_dir: str, seed: int, admission_only: bool):
    """
    Extracts information needed for the task of diagnoses prediction from the MIMIC dataset.
    The output data holds as labels all words of assigned diagnoses and all 3- and 4-digit codes.
    Creates 70/10/20 split over patients for train/val/test sets.
    """

    # set task name
    task_name = "DIA_PLUS"
    if admission_only:
        task_name = f"{task_name}_adm"

    # load dataframes
    mimic_dia_names = pd.read_csv(os.path.join(mimic_dir, "D_ICD_DIAGNOSES.csv"), dtype={"ICD9_CODE": str})
    mimic_diagnoses = pd.read_csv(os.path.join(mimic_dir, "DIAGNOSES_ICD.csv"))
    mimic_notes = pd.read_csv(os.path.join(mimic_dir, "NOTEEVENTS.csv"))
    mimic_admissions = pd.read_csv(os.path.join(mimic_dir, "ADMISSIONS.csv"))

    # filter notes
    mimic_notes = mimic_utils.filter_notes(mimic_notes, mimic_admissions, admission_text_only=admission_only)

    # only keep relevant columns
    mimic_diagnoses = mimic_diagnoses[['SUBJECT_ID', 'HADM_ID', 'ICD9_CODE']]

    # drop all rows without diagnoses codes
    mimic_diagnoses = mimic_diagnoses.dropna(how='any', subset=['ICD9_CODE', 'HADM_ID'], axis=0)

    # CREATE LABELS FOR DIAGNOSES NAMES

    # remove punctuation and split words of diagnoses descriptions
    mimic_dia_names["DIA_NAMES"] = mimic_dia_names.LONG_TITLE.str.replace('[{}]'.format(string.punctuation), '') \
        .str.lower().str.split()

    # remove stopwords and duplicates
    mimic_dia_names["DIA_NAMES"] = mimic_dia_names.DIA_NAMES.apply(
        lambda x: " ".join(set([word for word in x if word not in list(stopwords.words('english'))])))

    # CREATE LABELS FOR 3 DIGIT CODES

    # Truncate codes to 3 digits (preserve 3 digits for E and V codes aswell)
    mimic_diagnoses["SHORT_CODE"] = mimic_diagnoses.ICD9_CODE.astype(str)
    mimic_diagnoses.loc[
        mimic_diagnoses['SHORT_CODE'].str.startswith("V"), 'SHORT_CODE'] = mimic_diagnoses.SHORT_CODE.apply(
        lambda x: x[:4])
    mimic_diagnoses.loc[
        mimic_diagnoses['SHORT_CODE'].str.startswith("E"), 'SHORT_CODE'] = mimic_diagnoses.SHORT_CODE.apply(
        lambda x: x[:4])
    mimic_diagnoses.loc[(~mimic_diagnoses.SHORT_CODE.str.startswith("E")) & (
        ~mimic_diagnoses.SHORT_CODE.str.startswith("V")), 'SHORT_CODE'] = mimic_diagnoses.SHORT_CODE.apply(
        lambda x: x[:3])

    # CREATE LABELS FOR 4 DIGIT CODES

    # Truncate codes to 4 digits (preserve 4 digits for E and V codes aswell)
    mimic_diagnoses["LONG_CODE"] = mimic_diagnoses.ICD9_CODE.astype(str)
    mimic_diagnoses.loc[
        mimic_diagnoses['LONG_CODE'].str.startswith("V"), 'LONG_CODE'] = mimic_diagnoses.LONG_CODE.apply(
        lambda x: x[:5])
    mimic_diagnoses.loc[
        mimic_diagnoses['LONG_CODE'].str.startswith("E"), 'LONG_CODE'] = mimic_diagnoses.LONG_CODE.apply(
        lambda x: x[:5])
    mimic_diagnoses.loc[(~mimic_diagnoses.LONG_CODE.str.startswith("E")) & (
        ~mimic_diagnoses.LONG_CODE.str.startswith("V")), 'LONG_CODE'] = mimic_diagnoses.LONG_CODE.apply(
        lambda x: x[:4])

    # MERGE DESCRIPTION WITH ADMISSION CODES
    admissions_with_dia_names = pd.merge(mimic_diagnoses, mimic_dia_names[["ICD9_CODE", "DIA_NAMES"]],
                                         on="ICD9_CODE", how="left")
    admissions_with_dia_names["DIA_NAMES"] = admissions_with_dia_names.DIA_NAMES.fillna("")

    # GROUP CODES BY ADMISSION
    grouped_short_codes = admissions_with_dia_names.groupby(['HADM_ID'])['SHORT_CODE'].apply(" ".join).reset_index()
    grouped_long_codes = admissions_with_dia_names.groupby(['HADM_ID'])['LONG_CODE'].apply(" ".join).reset_index()
    grouped_dia_names = admissions_with_dia_names.groupby(['HADM_ID'])['DIA_NAMES'].apply(" ".join).reset_index()

    # COMBINE 3-DIGIT CODES, 4-DIGIT CODES AND DIAGNOSES NAMES

    # combine into one dataframe
    combined_df = reduce(lambda left, right: pd.merge(left, right, on=['HADM_ID'], how='outer'),
                         [grouped_short_codes, grouped_long_codes, grouped_dia_names])

    # combine into one column
    combined_df["LABELS"] = combined_df["SHORT_CODE"] + " " + combined_df["LONG_CODE"] + " " + combined_df["DIA_NAMES"]

    # remove duplicates, sort and join with comma
    combined_df["LABELS"] = combined_df.LABELS.str.split(" ").apply(lambda x: ",".join(sorted(set(x))))

    # merge discharge summaries into diagnoses table
    notes_diagnoses_df = pd.merge(combined_df[['HADM_ID', 'LABELS']], mimic_notes, how='inner', on='HADM_ID')

    # collect all possible tokens aka classes
    all_tokens = set()
    for i, row in notes_diagnoses_df.iterrows():
        for token in row.LABELS.split(","):
            all_tokens.add(token)

    mimic_utils.save_mimic_split_patient_wise(notes_diagnoses_df,
                                              label_column='LABELS',
                                              save_dir=save_dir,
                                              task_name=task_name,
                                              seed=seed)

    write_codes_to_file(sorted(all_tokens), save_dir)


def write_codes_to_file(icd_codes: List[str], data_path):
    # save ICD codes in an extra file
    with open(os.path.join(data_path, "ALL_DIAGNOSES_PLUS_CODES.txt"), "w", encoding="utf-8") as icd_file:
        icd_file.write(" ".join(icd_codes))


if __name__ == "__main__":
    args = mimic_utils.parse_args()
    dia_plus_mimic(args.mimic_dir, args.save_dir, args.seed, args.admission_only)
