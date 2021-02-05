import argparse
import os
import re
import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET
import csv


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--i2b2_dir', required=True)
    parser.add_argument('--save_dir', required=True)
    parser.add_argument('--header_dir', required=False)
    parser.add_argument('--admission_only', default=False)
    parser.add_argument('--seed', default=123, type=int)
    return parser.parse_args()


def filter_admission_text(i2b2_dir, header_dir, save_dir) -> pd.DataFrame:
    """
    Filter text information by section and only keep sections that are known on admission time.
    """

    # For both Set1 and Set2, extract section headers and their corresponding span information for the i2b2 data
    headings_set1 = header_info_extraction(header_dir, 'Set1')
    headings_set2 = header_info_extraction(header_dir, 'Set2')

    # Retrieve unique headers and corresponding section text's span information
    sec_text_span = unique_headings(headings_set1, headings_set2, save_dir)

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

    # List of mandatory sections
    mandatory_sections = ["CHIEF COMPLAINT: ", "IMPRESSION: ", "HISTORY OF PRESENT ILLNESS: ", "PAST MEDICAL HISTORY: ",
                          "REASON FOR VISIT: ", "MAJOR PROBLEMS: ", "HISTORY: ",
                          "PAST MEDICAL HISTORY AND SOCIAL HISTORY: ", "NARRATIVE HISTORY: ", "REASON FOR CONSULT: ",
                          "ATTENDING NOTE: ", "CHIEF COMPLAINT AND HISTORY OF PRESENT ILLNESS: ",
                          "REASON FOR ADMISSION: ", "PRESENT ILLNESS: "]

    filenames = []
    instances = []
    admission_text = ''
    first_rec = True

    # Combine the required sections of a clinical note under column "ADMISSION_TEXT"
    for index, row in sec_text_span.iterrows():
        if row['new_heading'] in admission_filter:

            if first_rec:
                filename = row['file']
                first_rec = False

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

            text = note[section_start:section_end].strip()
            replace_stars_underscores = re.compile(r"[-{3}+|***+|___+]")
            text = replace_stars_underscores.sub(" ", text).strip()
            combine_whitespace = re.compile(r"\s+")
            text = combine_whitespace.sub(" ", text).strip()

            # Drop the section text if blank
            if (text != ""):
                section_text = row['new_heading'].upper(
                ) + ': ' + text.replace('\n', ' ').strip()
            else:
                continue

            # Combine the different sections of a clinical note in one string
            if row['file'] == filename:
                admission_text = admission_text + '\n\n' + section_text.strip()
            else:
                # Drop the clinical note if none of the mandatory sections are included, else add to the output dataframe
                if any(section in admission_text for section in mandatory_sections):
                    instance = filename[:-
                    4], filename[:3], admission_text.strip()
                    instances.append(instance)

                admission_text = section_text
                filename = row['file']

    # instance represents a record of the output file including the ROW_ID, SUBJECT_ID and ADMISSION_TEXT
    instance = filename[:-4], filename[:3], admission_text.strip()
    instances.append(instance)

    adm_text_df = pd.DataFrame(
        instances, columns=['ROW_ID', 'SUBJECT_ID', 'ADMISSION_TEXT'])

    return adm_text_df


def unique_headings(headings_set1: pd.DataFrame, headings_set2: pd.DataFrame,
                    i2b2_headings_dict_path: str) -> pd.DataFrame:
    """
    Creates unique headings across all files.
    Creates columns which include information regarding the beginning and ending of a section's text.
    """

    # Combine section header information for both sets
    frames = [headings_set1, headings_set2]
    all_headings = pd.concat(frames)

    # Remove punctuation characters from section headers
    all_headings['heading'] = [
        (re.sub('[////\:*?"<>|#-=_]', ' ', elem)).strip() for elem in all_headings.heading]

    # Import the dictionary which maps existing section headers of both sets to new unique headers
    headings_dict_df = pd.read_csv(i2b2_headings_dict_path)
    mapping_dict = headings_dict_df.set_index('old_heading')['new_heading'].to_dict()

    # Use the imported dictionary to create column 'new heading' which includes unique headings
    all_headings['new_heading'] = all_headings['heading'].map(mapping_dict)
    all_headings['new_heading'] = np.where(all_headings['new_heading'].isnull(
    ), all_headings['heading'], all_headings['new_heading'])

    # Convert the data type of span columns from string to integer
    all_headings['span_end'] = all_headings['span_end'].astype(np.int64)
    all_headings['span_start'] = all_headings['span_start'].astype(np.int64)

    # Create the columns which include the character information for section beginning and ending
    all_headings['sec_start'] = all_headings['span_end'] + 1
    sorted_span = all_headings.groupby(["file"], sort=False).apply(
        lambda x: x.sort_values(["span_start"])).reset_index(drop=True)
    shifted = sorted_span.groupby(
        "file").shift(-1).drop(['heading', 'span_end', 'new_heading', 'sec_start'], axis=1)

    # Use lag functionality to get each section's last character position
    sec_text_span = sorted_span.join(
        shifted.rename(columns=lambda x: x + "_lag"))
    sec_text_span['span_start_lag'] = sec_text_span['span_start_lag'].fillna(
        -1)
    sec_text_span['span_start_lag'] = sec_text_span['span_start_lag'].astype(
        np.int64)
    sec_text_span['sec_end'] = sec_text_span['span_start_lag'] - 1

    return sec_text_span


def header_info_extraction(header_dir, set_name) -> pd.DataFrame:
    """
    Extracts the heading and span information using the section header data available for the i2b2 data
    """
    file_path = os.path.join(header_dir, set_name)

    # Names of clinical record files in alphabetical order:
    all_files = os.listdir(file_path)
    all_files.sort()

    instances = []

    for filename in all_files:

        # Read files:

        path = os.path.join(file_path, filename)
        file = open(path, 'r', encoding='utf-8').read()

        # Convert text to token sequences:
        token_sequences = [sentence.split() for sentence in file.split('\n')]

        # Assign the token position for the first word of the section heading
        word_pos = 4

        # Extract the heading, span information,filename and the corresponding set name into a Dataframe
        for seq in token_sequences:
            heading = seq[word_pos]
            span_start = seq[2]
            span_end = seq[3]
            while len(seq) - 1 != word_pos:
                word_pos = word_pos + 1
                heading = heading + ' ' + seq[word_pos]

            instance = heading.lower(), span_start, span_end, filename, set_name
            instances.append(instance)
            word_pos = 4

    headings_df = pd.DataFrame(
        instances, columns=['heading', 'span_start', 'span_end', 'file', 'set_name'])
    return headings_df


def save_i2b2_split_patient_wise(df, label_column, save_dir, task_name, seed, column_list=None):
    """
    Splits a i2b2 dataframe into 70/10/20 train, val, test with no patient occuring in more than one set.
    Uses ROW_ID as ID column and save to save_path.
    """
    if column_list is None:
        column_list = ["ID", "ADMISSION_TEXT", label_column]

    np.random.seed(seed)

    # Make a split per patient, so that no patients reoccur in one of the eval sets
    unique_patients = df.SUBJECT_ID.unique()

    np.random.shuffle(unique_patients)
    data_split = np.split(unique_patients, [int(.7 * len(unique_patients)),
                                            int(.8 * len(unique_patients))])

    # Use row id as general id
    df = df.rename(columns={'ROW_ID': 'ID'})

    # Create path to task data
    os.makedirs(save_dir, exist_ok=True)

    # Save splits to data folder
    for i, split_name in enumerate(["train", "val", "test"]):
        split_set = df[df.SUBJECT_ID.isin(data_split[i])].sample(
            frac=1, random_state=seed)[column_list]

        # lower case column names
        split_set.columns = map(str.lower, split_set.columns)

        split_set.to_csv(os.path.join(save_dir, "{}_{}.csv".format(task_name, split_name)),
                         index=False,
                         quoting=csv.QUOTE_ALL)
