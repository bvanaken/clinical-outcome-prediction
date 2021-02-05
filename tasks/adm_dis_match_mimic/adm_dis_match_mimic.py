import math
import os
import pandas as pd
from tasks import mimic_utils


def extract_section(df, section_heading):
    return df.TEXT.str.extract(r'(?i){}(.+?)\\n\\n[^(\\|\d|\.)]+?:'.format(section_heading))


def split_admission_discharge(mimic_dir: str, save_dir: str, seed: int):
    """
    Filter text information by section and only keep sections that are known on admission time.
    """

    # set task name
    task_name = "ADM_DIS_MATCH"

    # load dataframes
    mimic_notes = pd.read_csv(os.path.join(mimic_dir, "NOTEEVENTS.csv"),
                              usecols=["ROW_ID", "SUBJECT_ID", "HADM_ID", "CHARTDATE", "CATEGORY", "DESCRIPTION",
                                       "TEXT"])

    mimic_admissions = pd.read_csv(os.path.join(mimic_dir, "ADMISSIONS.csv"))

    # filter notes
    mimic_notes = mimic_utils.filter_notes(mimic_notes, mimic_admissions, admission_text_only=False)

    admission_sections = {
        "CHIEF_COMPLAINT": "chief complaint:",
        "PRESENT_ILLNESS": "present illness:",
        "MEDICAL_HISTORY": "medical history:",
        "MEDICATION_ADM": "medications on admission:",
        "ALLERGIES": ["allergy:", "allergies:"],
        "PHYSICAL_EXAM": ["physical exam:", "physical examination:"],
        "FAMILY_HISTORY": "family history:",
        "SOCIAL_HISTORY": "social history:"
    }

    discharge_sections = {
        "PROCEDURE": "procedure:",
        "MEDICATION_DIS": ["discharge medications:", "discharge medication:"],
        "DIAGNOSIS_DIS": ["discharge diagnosis:", "discharge diagnoses:"],
        "CONDITION": "discharge condition:",
        "PERTINENT_RESULTS": "pertinent results:",
        "HOSPITAL_COURSE": "hospital course:"
    }

    # replace linebreak indicators
    mimic_notes['TEXT'] = mimic_notes['TEXT'].str.replace(r"\n", r"\\n")

    # extract each section by regex
    for key in list(admission_sections.keys()) + list(discharge_sections.keys()):
        section = admission_sections[key] if key in admission_sections else discharge_sections[key]

        # handle multiple heading possibilities
        if isinstance(section, list):
            mimic_notes[key] = None
            for heading in section:
                mimic_notes.loc[mimic_notes[key].isnull(), key] = extract_section(mimic_notes, heading)
        else:
            mimic_notes[key] = extract_section(mimic_notes, section)

        mimic_notes[key] = mimic_notes[key].str.replace(r'\\n', r' ')
        mimic_notes[key] = mimic_notes[key].str.strip()
        mimic_notes[key] = mimic_notes[key].fillna("")
        mimic_notes[mimic_notes[key].str.startswith("[]")][key] = ""

    # filter notes with missing main admission information
    mimic_notes = mimic_notes[(mimic_notes.CHIEF_COMPLAINT != "") | (mimic_notes.PRESENT_ILLNESS != "") |
                              (mimic_notes.MEDICAL_HISTORY != "")]

    # filter notes with missing main information
    mimic_notes = mimic_notes[(mimic_notes.HOSPITAL_COURSE != "") | (mimic_notes.DIAGNOSIS_DIS != "")]

    # add section headers and combine into TEXT_ADMISSION
    mimic_notes = mimic_notes.assign(TEXT_ADMISSION="CHIEF COMPLAINT: " + mimic_notes.CHIEF_COMPLAINT.astype(str)
                                                    + '\n\n' +
                                                    "PRESENT ILLNESS: " + mimic_notes.PRESENT_ILLNESS.astype(str)
                                                    + '\n\n' +
                                                    "MEDICAL HISTORY: " + mimic_notes.MEDICAL_HISTORY.astype(str)
                                                    + '\n\n' +
                                                    "MEDICATION ON ADMISSION: " + mimic_notes.MEDICATION_ADM.astype(str)
                                                    + '\n\n' +
                                                    "ALLERGIES: " + mimic_notes.ALLERGIES.astype(str)
                                                    + '\n\n' +
                                                    "PHYSICAL EXAM: " + mimic_notes.PHYSICAL_EXAM.astype(str)
                                                    + '\n\n' +
                                                    "FAMILY HISTORY: " + mimic_notes.FAMILY_HISTORY.astype(str)
                                                    + '\n\n' +
                                                    "SOCIAL HISTORY: " + mimic_notes.SOCIAL_HISTORY.astype(str))

    # add section headers and combine into TEXT_DISCHARGE
    mimic_notes = mimic_notes.assign(
        TEXT_DISCHARGE="MAJOR SURGICAL / INVASIVE PROCEDURE: " + mimic_notes.PROCEDURE.astype(str)
                       + '\n\n' +
                       "PERTINENT RESULTS: " + mimic_notes.PERTINENT_RESULTS.astype(str)
                       + '\n\n' +
                       "HOSPITAL COURSE: " + mimic_notes.HOSPITAL_COURSE.astype(str)
                       + '\n\n' +
                       "DISCHARGE MEDICATIONS: " + mimic_notes.MEDICATION_DIS.astype(str)
                       + '\n\n' +
                       "DISCHARGE DIAGNOSES: " + mimic_notes.DIAGNOSIS_DIS.astype(str)
                       + '\n\n' +
                       "DISCHARGE CONDITION: " + mimic_notes.CONDITION.astype(str))

    mimic_utils.save_mimic_split_patient_wise(
        df=mimic_notes[['ROW_ID', 'SUBJECT_ID', 'TEXT_ADMISSION', 'TEXT_DISCHARGE']],
        label_column=None,
        column_list=['ID', 'TEXT_ADMISSION', 'TEXT_DISCHARGE'],
        save_dir=save_dir,
        task_name=task_name,
        seed=seed)


def create_pretraining_file(save_dir):
    task_name = "ADM_DIS_MATCH"

    # Only use MIMIC train set for pretraining task
    base_df = pd.read_csv(f"{os.path.join(save_dir, task_name)}_train.csv")

    # Create val set
    # 1. Shuffle
    base_df = base_df.sample(frac=1)

    # 2. Define split size
    val_split = 0.005
    val_size = math.ceil(len(base_df) * val_split)

    # 3. Split
    splits = {
        "train": base_df.iloc[val_size:, :],
        "val": base_df.iloc[:val_size, :]
    }

    for split_name in splits:
        file_content = ""
        for j, row in splits[split_name].iterrows():
            file_content += row["text_admission"].replace("\n", " ")
            file_content += "\n[SEP]\n"
            file_content += row["text_discharge"].replace("\n", " ")
            file_content += "\n\n"

        file_name = f"{os.path.join(save_dir, task_name)}_{split_name}.txt"
        with open(file_name, "w", encoding="utf-8") as write_file:
            write_file.write(file_content)


if __name__ == "__main__":
    args = mimic_utils.parse_args()
    split_admission_discharge(args.mimic_dir, args.save_dir, args.seed)

    create_pretraining_file(args.save_dir)
