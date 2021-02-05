import math

from tasks import i2b2_utils
import os
import pandas as pd
import numpy as np
import re
import xml.etree.ElementTree as ET

STARS_UNDERSCORE_PATTERN = re.compile(r"[-{3}+|***+|___+]")
WHITESPACE_PATTERN = re.compile(r"\s+")


def convert_i2b2_risk_factors_to_admission_only(i2b2_dir: str, header_dir: str, save_dir: str, seed: int,
                                                i2b2_headings_dict_path: str):
    """
    Filter text information by section and only keep sections that are known on admission and discharge time.
    """

    # set task name
    task_name = "ADM_DIS_MATCH_I2B2"

    # For both Set1 and Set2, extract section headers and their corresponding span information for the i2b2 data
    headings_set1 = i2b2_utils.header_info_extraction(header_dir, 'Set1')
    headings_set2 = i2b2_utils.header_info_extraction(header_dir, 'Set2')

    # Retrieve unique headers and corresponding section text's span information
    sec_text_span = i2b2_utils.unique_headings(
        headings_set1, headings_set2, i2b2_headings_dict_path)

    # List of section headers under admission group
    admission_filter = ["chief complaint", "physical examination", "impression", "history of present illness",
                        "past medical history", "allergies", "review of systems", "family history", "social history",
                        "reason for visit", "past surgical history", "current medications",
                        "social history and family history", "medications on admission", "major problems", "history",
                        "past medical history and social history",
                        "narrative history", "reason for consult", "changes to allergies", "attending note",
                        "therapy rendered course in ed", "chief complaint and history of present illness",
                        "transfer medications", "interim history", "reason for admission", "other problems",
                        "labs on admission", "present illness", "medications", "problems", "interval history", "habits",
                        "medications at home"]

    # List of section headers under discharge group
    discharge_filter = ["assessment and plan", "impression and plan", "lab data", "plan", "assessment",
                        "emergency department course", "vital signs", "interval history", "final diagnosis",
                        "diagnosis", "disposition including condition upon discharge", "disposition",
                        "laboratory evaluation", "recommendations", "health maintenance", "radiology", "procedures",
                        "medications at home", "change in therapies and renewals",
                        "medications confirmed", "imaging", "radiology", "results", "changes to medications this visit",
                        "note", "findings", "perceptor note", "primary diagnosis", "consultations including pcp",
                        "allergies adverse reactions", "assessment and recommendations", "appointments",
                        "review comments", "condition on discharge", "secondary diagnosis", "dispo",
                        "medical decision making plan", "final diagnoses"]

    # List of mandatory sections for admission group
    adm_mandatory_sections = ["CHIEF COMPLAINT: ", "IMPRESSION: ", "HISTORY OF PRESENT ILLNESS: ",
                              "PAST MEDICAL HISTORY: ", "REASON FOR VISIT: ", "MAJOR PROBLEMS: ", "HISTORY: ",
                              "PAST MEDICAL HISTORY AND SOCIAL HISTORY: ", "NARRATIVE HISTORY: ",
                              "REASON FOR CONSULT: ", "ATTENDING NOTE: ",
                              "CHIEF COMPLAINT AND HISTORY OF PRESENT ILLNESS: ", "REASON FOR ADMISSION: ",
                              "PRESENT ILLNESS: "]

    # List of mandatory sections for discharge group
    dis_mandatory_sections = ["ASSESSMENT AND PLAN", "IMPRESSION AND PLAN", "PLAN", "ASSESSMENT",
                              "EMERGENCY DEPARTMENT COURSE", "FINAL DIAGNOSIS", "DIAGNOSIS",
                              "DISPOSITION INCLUDING CONDITION UPON DISCHARGE", "DISPOSITION", "RECOMMENDATIONS",
                              "PROCEDURES", "CHANGE IN THERAPIES AND RENEWALS", "RESULTS", "FINDINGS",
                              "PRIMARY DIAGNOSIS", "CONSULTATIONS INCLUDING PCP", "ASSESSMENT AND RECOMMENDATIONS",
                              "CONDITION ON DISCHARGE", "DISPO", "MEDICAL DECISION MAKING PLAN", "FINAL DIAGNOSES"]

    # Combined list of section headers for admission and discharge group
    combined_adm_dis_filter = [y for x in [
        admission_filter, discharge_filter] for y in x]

    filenames = []
    instances = []
    admission_text = ''
    discharge_text = ''
    first_rec = True

    # Combine the required sections of a clinical note under column "ADMISSION_TEXT"
    for index, row in sec_text_span.iterrows():
        if (row['new_heading'] in combined_adm_dis_filter):

            if first_rec:
                filename = row['file']
                first_rec = False

            adm_text = True if row['new_heading'] in admission_filter else False
            dis_text = True if row['new_heading'] in discharge_filter else False

            # Extract the corresponding XML file for i2b2 data
            XML_FOLDER = 'training-RiskFactors-Complete-' + row['set_name']
            XML_FILE_PATH = os.path.join(i2b2_dir, XML_FOLDER)

            f_name = row['file'][:-3] + "xml"
            file_full_path = os.path.join(XML_FILE_PATH, f_name)

            # Parse the XML file to extract the text information
            tree = ET.parse(file_full_path)
            root = tree.getroot()
            note = root[0].text

            # Retrieve the beginning and end information for sections
            section_start = row['sec_start']
            if row['sec_end'] < 0:
                section_end = len(note)
            else:
                section_end = row['sec_end']
            filenames.append(row['file'])

            # Clean text by replacing unwanted characters and combining whitespace
            text = note[section_start:section_end].strip()
            text = STARS_UNDERSCORE_PATTERN.sub(" ", text).strip()
            text = WHITESPACE_PATTERN.sub(" ", text).strip()

            # Drop the section text if blank
            if (text != ""):
                section_text = row['new_heading'].upper(
                ) + ': ' + text.replace('\n', ' ').strip()
            else:
                continue

            # Combine the different sections of a clinical note in one string
            if row['file'] == filename:
                if adm_text:
                    admission_text = admission_text + '\n\n' + section_text.strip()
                if dis_text:
                    discharge_text = discharge_text + '\n\n' + section_text.strip()
            else:
                # Drop the clinical note if none of the mandatory sections are included, else add to the output dataframe
                if any(section in admission_text for section in
                       adm_mandatory_sections or dis_mandatory_sections) and discharge_text.strip() != '' and admission_text.strip() != '':
                    instance = filename[:-4], filename[:3], admission_text.strip(
                    ), discharge_text.strip()
                    instances.append(instance)

                if adm_text:
                    admission_text = section_text
                    discharge_text = ''
                if dis_text:
                    discharge_text = section_text
                    admission_text = ''

                filename = row['file']

    # instance represents a record of the output file including the ROW_ID, SUBJECT_ID, ADMISSION_TEXT and DISCHARGE_TEXT
    instance = filename[:-
    4], filename[:3], admission_text.strip(), discharge_text.strip()
    instances.append(instance)

    adm_text_df = pd.DataFrame(
        instances, columns=['ROW_ID', 'SUBJECT_ID', 'ADMISSION_TEXT', 'DISCHARGE_TEXT'])

    i2b2_utils.save_i2b2_split_patient_wise(
        df=adm_text_df[['ROW_ID', 'SUBJECT_ID',
                        'ADMISSION_TEXT', 'DISCHARGE_TEXT']],
        label_column=None,
        column_list=['ID', 'ADMISSION_TEXT', 'DISCHARGE_TEXT'],
        save_dir=save_dir,
        task_name=task_name, seed=seed)


def sentence_to_section(i2b2_dir: str) -> pd.DataFrame:
    """
    Combines sentences belonging to same section
    """

    # Load the i2b2 data including predicted section headers
    file_path = os.path.join(i2b2_dir, 'i2b2_section_predictions.csv')
    all_i2b2_notes = pd.read_csv(file_path)

    # Dictionary for different section headers
    header_dict = {
        'adm_med_text': {"header": "admission medications", "sentence": ''},
        'cc_text': {"header": "chief complaint", 'sentence': ''},
        'exam_text': {"header": "examination", 'sentence': ''},
        'hpi_text': {"header": "history of present illness", 'sentence': ''},
        'pmh_text': {"header": "past medical history", 'sentence': ''},
        'allergies_text': {"header": "allergies", 'sentence': ''},
        'ros_text': {"header": "review of systems", 'sentence': ''},
        'family_hx_text': {"header": "family history", 'sentence': ''},
        'Per_shx_text': {"header": "personal and social history", 'sentence': ''},
        'shx_text': {"header": "social history and family history", 'sentence': ''},
        'narrative_text': {"header": "narrative history", 'sentence': ''},
        'cc_hpi_text': {"header": "chief complaint and history of present illness", 'sentence': ''},
        'ap_text': {"header": "assessment and plan", 'sentence': ''},
        'lab_text': {"header": "lab data", 'sentence': ''},
        'ed_text': {"header": "emergency department course", 'sentence': ''},
        'interval_text': {"header": "interval history", 'sentence': ''},
        'dis_med_text': {"header": "discharge medications", 'sentence': ''},
        'findings_text': {"header": "findings", 'sentence': ''},
        'dispo_text': {"header": "disposition including condition upon discharge", 'sentence': ''},
        'health_text': {"header": "health maintenance", 'sentence': ''},
        'pro_text': {"header": "procedures", 'sentence': ''},
        'note_text': {"header": "note", 'sentence': ''},
        'appointments_text': {"header": "appointments", 'sentence': ''},
        'review_text': {"header": "review comments", 'sentence': ''},
    }

    first_rec = True
    instances = []
    # Combine the sentences belonging to same section under "Text" column
    for index, row in all_i2b2_notes.iterrows():
        if first_rec:
            filename = row['Filename']
            first_rec = False
        if filename == row['Filename']:
            for key in header_dict:
                if row['Header'] == header_dict[key]['header']:
                    header_dict[key]['sentence'] = header_dict[key]['sentence'] + row['Sentence']
        else:
            for key in header_dict:
                if header_dict[key]['sentence'] != '':
                    instance = (filename, header_dict[key]['header'], header_dict[key]['sentence'])
                    instances.append(instance)
            filename = row['Filename']
            for key in header_dict:
                if header_dict[key]['header'] == row['Header']:
                    header_dict[key]['sentence'] = row['Sentence']
                else:
                    header_dict[key]['sentence'] = ''

    for key in header_dict:
        if header_dict[key]['sentence'] != '':
            # instance represents a record of the output dataframe including the Filename, Section Header and  corresponding TEXT to that header
            instance = (filename, header_dict[key]['header'], header_dict[key]['sentence'])
            instances.append(instance)
    df_sen_to_sec = pd.DataFrame(
        instances, columns=['Filename', 'Header', 'Text'])
    return df_sen_to_sec


def create_adm_dis_pretraining_task(i2b2_dir: str, save_dir: str, seed: int):
    """
    Filter text information by predicted section and only keep sections that are known on admission and discharge time.
    """
    # set task name
    task_name = "ADM_DIS_MATCH_I2B2_PRED"

    # Retrieve dataframe with sentences belonging to same section combined
    df_sen_to_sec = sentence_to_section(i2b2_dir)

    # List of section headers under admission group
    admission_filter = ["admission_medications", "chief complaint", "examination", "history of present illness",
                        "past medical history", "allergies", "review of systems", "family history",
                        "personal and social history", "social history and family history", "narrative history",
                        "chief complaint and history of present illness"]

    # List of section headers under discharge group
    discharge_filter = ["assessment and plan", "lab data", "emergency department course", "interval history",
                        "discharge medications",
                        "findings", "disposition including condition upon discharge", "health maintenance",
                        "procedures",
                        "note", "appointments", "review comments"]

    # List of mandatory sections for admission group
    adm_mandatory_sections = ["CHIEF COMPLAINT: ", "HISTORY OF PRESENT ILLNESS: ", "PAST MEDICAL HISTORY: ",
                              "PAST MEDICAL HISTORY AND SOCIAL HISTORY: ", "NARRATIVE HISTORY: ",
                              "CHIEF COMPLAINT AND HISTORY OF PRESENT ILLNESS: "]

    # List of mandatory sections for discharge group
    dis_mandatory_sections = ["ASSESSMENT AND PLAN: ", "EMERGENCY DEPARTMENT COURSE: ",
                              "DISPOSITION INCLUDING CONDITION UPON DISCHARGE: ", "PROCEDURES: ", "FINDINGS: "]

    first_rec = True
    instances = []
    admission_text = ''
    discharge_text = ''

    for index, row in df_sen_to_sec.iterrows():

        if first_rec:
            filename = row['Filename']
            first_rec = False

        adm_text = True if row['Header'] in admission_filter else False
        dis_text = True if row['Header'] in discharge_filter else False

        # Clean text by replacing unwanted characters and combining whitespace
        section_text = row['Header'].upper() + ': ' + row['Text'].lower().strip()
        section_text = STARS_UNDERSCORE_PATTERN.sub(" ", section_text).strip()
        section_text = WHITESPACE_PATTERN.sub(" ", section_text).strip()

        # Combine the different sections of a clinical note in one string
        if row['Filename'] == filename:
            if adm_text:
                admission_text = admission_text + '\n\n' + \
                                 section_text.replace('\n', ' ').strip()
            if dis_text:
                discharge_text = discharge_text + '\n\n' + \
                                 section_text.replace('\n', ' ').strip()

        else:
            # Drop the clinical note if none of the mandatory sections are included, else add to the output dataframe
            if any(section in admission_text for section in
                   adm_mandatory_sections or dis_mandatory_sections) and discharge_text.strip() != '' and admission_text.strip() != '':
                instance = filename, filename[5:], admission_text.strip(), discharge_text.strip()
                instances.append(instance)
            if adm_text:
                admission_text = section_text.replace('\n', ' ').strip()
                discharge_text = ''
            if dis_text:
                discharge_text = section_text.replace('\n', ' ').strip()
                admission_text = ''
            filename = row['Filename']

    # instance represents a record of the output file including the ROW_ID, SUBJECT_ID, ADMISSION_TEXT and DISCHARGE_TEXT
    instance = filename, filename[5:], admission_text.strip(), discharge_text.strip()
    instances.append(instance)

    adm_text_pred_df = pd.DataFrame(
        instances, columns=['ROW_ID', 'SUBJECT_ID', 'ADMISSION_TEXT', 'DISCHARGE_TEXT'])

    # shuffle instances
    adm_text_pred_df = adm_text_pred_df.sample(frac=1)

    # define split size
    val_split = 0.15
    val_size = math.ceil(len(adm_text_pred_df) * val_split)

    # split into train and val
    splits = {
        "train": adm_text_pred_df[val_size:],
        "val": adm_text_pred_df[:val_size]
    }

    for split_name in splits:
        file_name = f"{os.path.join(save_dir, task_name)}_{split_name}.csv"
        splits[split_name].to_csv(file_name, index=False)


def create_pretraining_files(save_dir, task_name):
    for i, split_name in enumerate(["train", "val"]):
        split_df = pd.read_csv(f"{os.path.join(save_dir, task_name)}_{split_name}.csv", dtype={
            'ID': 'str', 'ADMISSION_TEXT': 'str', 'DISCHARGE_TEXT': 'str'})

        # replace np.nan with empty string
        split_df = split_df.replace(np.nan, '', regex=True)

        file_content = ""
        for j, row in split_df.iterrows():
            file_content += row["ADMISSION_TEXT"].replace("\n", " ")
            file_content += "\n[SEP]\n"
            file_content += row["DISCHARGE_TEXT"].replace("\n", " ")
            file_content += "\n\n"

        file_name = f"{os.path.join(save_dir, task_name)}_{split_name}.txt"
        with open(file_name, "w", encoding="utf-8") as write_file:
            write_file.write(file_content)


if __name__ == "__main__":
    args = i2b2_utils.parse_args()

    create_adm_dis_pretraining_task(args.i2b2_dir, args.save_dir, args.seed)
    create_pretraining_files(
        args.save_dir, task_name='ADM_DIS_MATCH_I2B2_PRED')
