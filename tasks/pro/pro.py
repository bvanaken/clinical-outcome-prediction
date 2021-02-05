import os
from typing import List
import sys

sys.path.append('.')
from tasks import mimic_utils
import pandas as pd


def pro_groups_3_digits_mimic(mimic_dir: str, save_dir: int, seed: int, admission_only: bool):
    """
    Extracts information needed for the task from the MIMIC dataset. Namely "TEXT" column from NOTEEVENTS.csv and
    "ICD9_CODE" from PROCEDURES_ICD.csv. Divide all ICD9 codes' first two digits and group them per admission into
    column "SHORT_CODES".
    Creates 70/10/20 split over patients for train/val/test sets.
    """

    # set task name
    task_name = "PRO_GROUPS_3_DIGITS"

    if admission_only:
        task_name = f"{task_name}_adm"

    # load dataframes
    mimic_procedures = pd.read_csv(
        os.path.join(mimic_dir, "PROCEDURES_ICD.csv"))
    mimic_notes = pd.read_csv(os.path.join(mimic_dir, "NOTEEVENTS.csv"))
    mimic_admissions = pd.read_csv(os.path.join(mimic_dir, "ADMISSIONS.csv"))

    # filter notes
    mimic_notes = mimic_utils.filter_notes(
        mimic_notes, mimic_admissions, admission_text_only=admission_only)

    # only keep relevant columns
    mimic_procedures = mimic_procedures[['SUBJECT_ID', 'HADM_ID', 'ICD9_CODE']]

    # drop all rows without procedure codes
    mimic_procedures = mimic_procedures.dropna(
        how='any', subset=['ICD9_CODE'], axis=0)

    # convert data type of ICD9_CODE from integer to string
    mimic_procedures.ICD9_CODE = mimic_procedures.ICD9_CODE.astype(str)

    # create column SHORT_CODE including first 2 digits of ICD9 code
    mimic_procedures["SHORT_CODE"] = mimic_procedures.ICD9_CODE.astype(
        str).str[:3]

    icd9_codes = mimic_procedures.SHORT_CODE.unique().tolist()

    # remove duplicated code groups per admission
    mimic_procedures = mimic_procedures.drop_duplicates(
        ["HADM_ID", "SHORT_CODE"])

    grouped_codes = mimic_procedures.groupby(['HADM_ID', 'SUBJECT_ID'])['SHORT_CODE'].apply(
        lambda d: ",".join(d.astype(str))).reset_index()

    # rename column
    grouped_codes = grouped_codes.rename(columns={'SHORT_CODE': 'SHORT_CODES'})

    # merge discharge summaries into procedures table
    notes_procedures_df = pd.merge(
        grouped_codes[['HADM_ID', 'SHORT_CODES']], mimic_notes, how='inner', on='HADM_ID')

    mimic_utils.save_mimic_split_patient_wise(notes_procedures_df,
                                              label_column='SHORT_CODES',
                                              save_dir=save_dir,
                                              task_name=task_name,
                                              seed=seed)

    # save file with all occuring codes
    write_icd_codes_to_file(icd9_codes, save_dir)


def write_icd_codes_to_file(icd_codes: List[str], data_path):
    # save ICD codes in an extra file
    with open(os.path.join(data_path, "ALL_3_DIGIT_PRO_CODES.txt"), "w", encoding="utf-8") as icd_file:
        icd_file.write(" ".join(icd_codes))


if __name__ == "__main__":
    args = mimic_utils.parse_args()
    pro_groups_3_digits_mimic(
        args.mimic_dir, args.save_dir, args.seed, args.admission_only)
