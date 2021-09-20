import os
import re
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
    notes_with_expire_flag = pd.merge(mimic_notes, mimic_admissions[["HADM_ID", "HOSPITAL_EXPIRE_FLAG"]], how="left",
                                      on="HADM_ID")

    # drop all rows without hospital expire flag
    notes_with_expire_flag = notes_with_expire_flag.dropna(how='any', subset=['HOSPITAL_EXPIRE_FLAG'], axis=0)

    # filter out written out death indications
    notes_with_expire_flag = remove_mentions_of_patients_death(notes_with_expire_flag)

    mimic_utils.save_mimic_split_patient_wise(notes_with_expire_flag,
                                              label_column='HOSPITAL_EXPIRE_FLAG',
                                              save_dir=save_dir,
                                              task_name=task_name,
                                              seed=seed)


def remove_mentions_of_patients_death(df: pd.DataFrame):
    """
    Some notes contain mentions of the patient's death such as 'patient deceased'. If these occur in the sections PHYSICAL EXAM and MEDICATION ON ADMISSION, we can simply remove the mentions, because the conditions are not further elaborated in these sections. However, if the mentions occur in any other section, such as CHIEF COMPLAINT, we want to remove the whole sample, because the patient's passing if usually closer described in the text and an outcome prediction does not make sense in these cases.
    """

    death_indication_in_special_sections = re.compile(
        r"((?:PHYSICAL EXAM|MEDICATION ON ADMISSION):[^\n\n]*?)((?:patient|pt)?\s+(?:had\s|has\s)?(?:expired|died|passed away|deceased))",
        flags=re.IGNORECASE)

    death_indication_in_all_other_sections = re.compile(
        r"(?:patient|pt)\s+(?:had\s|has\s)?(?:expired|died|passed away|deceased)", flags=re.IGNORECASE)

    # first remove mentions in sections PHYSICAL EXAM and MEDICATION ON ADMISSION
    df['TEXT'] = df['TEXT'].replace(death_indication_in_special_sections, r"\1", regex=True)

    # if mentions can be found in any other section, remove whole sample
    df = df[~df['TEXT'].str.contains(death_indication_in_all_other_sections)]

    # remove other samples with obvious death indications
    df = df[~df['TEXT'].str.contains("he expired", flags=re.IGNORECASE)]  # does also match 'she expired'
    df = df[~df['TEXT'].str.contains("pronounced expired", flags=re.IGNORECASE)]
    df = df[~df['TEXT'].str.contains("time of death", flags=re.IGNORECASE)]

    return df


if __name__ == "__main__":
    args = mimic_utils.parse_args()
    mp_in_hospital_mimic(args.mimic_dir, args.save_dir, args.seed, args.admission_only)
