import os
import sys

sys.path.append('.')

from tasks import mimic_utils
import pandas as pd


def mp_in_hospital_mimic(mimic_dir: str, save_dir: str, seed: int, admission_only: bool):
    """
    Extracts information needed for the task from the MIMIC dataset. Namely "TEXT" column from NOTEEVENTS.csv and
    "HOSPITAL_EXPIRE_FLAG" from ADMISSIONS.csv. Filters specific admission sections for often occuring signal words.
    Creates 70/10/20 split over patients for train/val/test sets.
    """

    # set task name
    task_name = "MP_IN"
    if admission_only:
        task_name = f"{task_name}_adm"

    # load dataframes
    mimic_notes = pd.read_csv(os.path.join(mimic_dir, "NOTEEVENTS.csv"))
    mimic_admissions = pd.read_csv(os.path.join(mimic_dir, "ADMISSIONS.csv"))

    # filter notes
    mimic_notes = mimic_utils.filter_notes(mimic_notes, mimic_admissions, admission_text_only=admission_only)

    # append HOSPITAL_EXPIRE_FLAG to notes
    notes_expire_flag = pd.merge(mimic_notes, mimic_admissions[["HADM_ID", "HOSPITAL_EXPIRE_FLAG"]], how="left",
                                 on="HADM_ID")

    # drop all rows without hospital expire flag
    notes_expire_flag = notes_expire_flag.dropna(how='any', subset=['HOSPITAL_EXPIRE_FLAG'], axis=0)

    # filter out written out death indications
    notes_expire_flag['TEXT'] = notes_expire_flag['TEXT'].str.replace('patient died', '')
    notes_expire_flag['TEXT'] = notes_expire_flag['TEXT'].str.replace('patient deceased', '')
    notes_expire_flag['TEXT'] = notes_expire_flag['TEXT'].str.replace('\ndeceased\n', '\n')

    mimic_utils.save_mimic_split_patient_wise(notes_expire_flag,
                                              label_column='HOSPITAL_EXPIRE_FLAG',
                                              save_dir=save_dir,
                                              task_name=task_name,
                                              seed=seed)


if __name__ == "__main__":
    args = mimic_utils.parse_args()
    mp_in_hospital_mimic(args.mimic_dir, args.save_dir, args.seed, args.admission_only)
