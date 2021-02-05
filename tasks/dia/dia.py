import os
import sys

sys.path.append('.')
from typing import List

from tasks import mimic_utils
import pandas as pd


def dia_groups_3_digits_mimic(mimic_dir: str, save_dir: int, seed: int, admission_only: bool):
    """
    Extracts information needed for the task from the MIMIC dataset. Namely "TEXT" column from NOTEEVENTS.csv and
    "ICD9_CODE" from DIAGNOSES_ICD.csv. Divide all ICD9 codes' first three digits and group them per admission into
    column "SHORT_CODES".
    Creates 70/10/20 split over patients for train/val/test sets.
    """

    # set task name
    task_name = "DIA_GROUPS_3_DIGITS"

    if admission_only:
        task_name = f"{task_name}_adm"

    # load dataframes
    mimic_diagnoses = pd.read_csv(os.path.join(mimic_dir, "DIAGNOSES_ICD.csv"))
    mimic_notes = pd.read_csv(os.path.join(mimic_dir, "NOTEEVENTS.csv"))
    mimic_admissions = pd.read_csv(os.path.join(mimic_dir, "ADMISSIONS.csv"))

    # filter notes
    mimic_notes = mimic_utils.filter_notes(
        mimic_notes, mimic_admissions, admission_text_only=admission_only)

    # only keep relevant columns
    mimic_diagnoses = mimic_diagnoses[['SUBJECT_ID', 'HADM_ID', 'ICD9_CODE']]

    # drop all rows without diagnosis codes
    mimic_diagnoses = mimic_diagnoses.dropna(
        how='any', subset=['ICD9_CODE'], axis=0)

    # create column SHORT_CODE including first 3 digits of ICD9 code
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

    # remove duplicated code groups per admission
    mimic_diagnoses = mimic_diagnoses.drop_duplicates(
        ["HADM_ID", "SHORT_CODE"])

    # store all ICD codes for vectorization
    icd9_codes = mimic_diagnoses.SHORT_CODE.unique().tolist()

    grouped_codes = mimic_diagnoses.groupby(['HADM_ID', 'SUBJECT_ID'])['SHORT_CODE'].apply(
        lambda d: ",".join(d.astype(str))).reset_index()

    # rename column
    grouped_codes = grouped_codes.rename(columns={'SHORT_CODE': 'SHORT_CODES'})

    # merge discharge summaries into diagnosis table
    notes_diagnoses_df = pd.merge(
        grouped_codes[['HADM_ID', 'SHORT_CODES']], mimic_notes, how='inner', on='HADM_ID')

    mimic_utils.save_mimic_split_patient_wise(notes_diagnoses_df,
                                              label_column='SHORT_CODES',
                                              save_dir=save_dir,
                                              task_name=task_name,
                                              seed=seed)

    # save file with all occuring codes
    write_icd_codes_to_file(icd9_codes, save_dir)


def write_icd_codes_to_file(icd_codes: List[str], data_path):
    # save ICD codes in an extra file
    with open(os.path.join(data_path, "ALL_3_DIGIT_DIA_CODES.txt"), "w", encoding="utf-8") as icd_file:
        icd_file.write(" ".join(icd_codes))


if __name__ == "__main__":
    args = mimic_utils.parse_args()
    dia_groups_3_digits_mimic(
        args.mimic_dir, args.save_dir, args.seed, args.admission_only)
