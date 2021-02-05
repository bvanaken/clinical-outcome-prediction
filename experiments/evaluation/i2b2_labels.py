import os

import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET


def load_i2b2_test_examples(path):
    return pd.read_csv(path,
                       sep='\t',
                       encoding='utf-8')


def load_i2b2_train_examples(path):
    df = pd.read_csv(path, encoding='utf-8')
    df['True'] = df['Section Class']
    df['patient_id'] = df['FILENAME'].apply(lambda x: x[:-4])
    df['Sentence'] = df['TEXT']
    return df[['patient_id', 'Sentence', 'True']]


def filter_patients_with_admission(df, admission_sections):
    new_df = df[df['True'].isin(admission_sections)]
    return new_df


def extract_patient_id(df):
    df['patient_id'] = df['Location'].apply(lambda x: x[-10:-4])
    return df


def gather_patient_text(admission_df, patient_ids):
    new_df = pd.DataFrame(data=np.zeros((len(patient_ids), 3)),
                          columns=['patient_id',
                                   'text',
                                   'labels'
                                   ])

    for idx, pid in enumerate(patient_ids):
        text_snippets = admission_df.loc[admission_df.patient_id == pid, 'Sentence']
        patient_text = text_snippets.str.cat(sep=' ')
        new_df.loc[idx, 'patient_id'] = pid
        new_df.loc[idx, 'text'] = patient_text
    return new_df


def get_patient_labels(df, text_icd_map, xml_path):
    for idx, row in df.iterrows():
        pid = row.patient_id

        tree = ET.parse(os.path.join(xml_path, pid + ".xml"))
        root = tree.getroot()

        labels = list()
        no_time = list()
        out_range = list()

        for xml_key in root[1]:

            if 'time' in xml_key.attrib and xml_key.tag != 'MEDICATION':
                if xml_key.attrib['time'] in ['before DCT', 'during DCT', 'after DCT']:
                    labels.append(text_icd_map[xml_key.tag])
                else:
                    out_range.append(xml_key.tag)
            else:
                no_time.append(xml_key.tag)

        df.loc[idx, 'labels'] = ','.join(set(labels))

    return df


def get_patients_with_label(df):
    tmp = df['labels'].apply(lambda x: len(x))
    return df[tmp != 0]


def get_arguments():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir")
    parser.add_argument("--i2b2_section_labels_train")
    parser.add_argument("--i2b2_section_labels_test")
    # i2b2_risk_factors_train_dir: Directory containing files from 'testing-RiskFactors-Complete'
    parser.add_argument("--i2b2_risk_factors_train_dir")
    # i2b2_risk_factors_test_dir: Directory containing files from 'training-RiskFactors-Complete-Set1' and
    # 'training-RiskFactors-Complete-Set2'
    parser.add_argument("--i2b2_risk_factors_test_dir")

    return parser.parse_args()


if __name__ == '__main__':
    args = get_arguments()

    text_icd_map = {'HYPERTENSION': '401',
                    'HYPERLIPIDEMIA': '272',
                    'CAD': '414',
                    'DIABETES': '250',
                    'OBESE': '278'}

    """
        ALL SECTIONS found in document
        ['assessment and plan' 'history of present illness' 'past medical history'
         'allergies' 'personal and social history' 'examination' 'lab data'
         'chief complaint' 'admission medications' 'findings'
         'chief complaint and history of present illness' 'interval history'
         'family history' 'review of systems' 'emergency department course'
         'disposition and condition on discharge' 'procedures'
         'discharge medications' 'social history and family history'
         'narrative history' 'note']
    """

    admission_sections = [
        'history of present illness',
        'past medical history',
        'allergies',
        'personal and social history',
        'chief complaint',
        'admission medications',
        'chief complaint and history of present illness',
        'social history and family history',
        'narrative history',
        'note',
        'family history',
        'examination',
        'review of systems'
    ]

    # TRAIN_DATA
    train_data = load_i2b2_train_examples(args.i2b2_section_labels_train)
    train_patients_with_admission = filter_patients_with_admission(train_data, admission_sections)
    all_train_patients = train_patients_with_admission['patient_id'].unique()
    train_patient_admission_text = gather_patient_text(train_patients_with_admission, all_train_patients)
    train_patient_admission_state = get_patient_labels(train_patient_admission_text, text_icd_map,
                                                       args.i2b2_risk_factors_train_dir)
    train_patient_admission_state_with_label = get_patients_with_label(train_patient_admission_state)

    # TEST DATA

    i2b2_data = load_i2b2_test_examples(args.i2b2_section_labels_test)
    i2b2_data = extract_patient_id(i2b2_data)
    patients_with_admission = filter_patients_with_admission(i2b2_data, admission_sections)
    all_patients = patients_with_admission['patient_id'].unique()
    patient_admission_text = gather_patient_text(patients_with_admission, all_patients)
    patient_admission_state = get_patient_labels(patient_admission_text, text_icd_map, args.i2b2_risk_factors_test_dir)
    patient_admission_state_with_label = get_patients_with_label(patient_admission_state)

    total_patient_admission_state_with_label = pd.concat([patient_admission_state_with_label,
                                                          train_patient_admission_state_with_label])

    total_patient_admission_state_with_label.to_csv(
        os.path.join(args.output_dir, 'i2b2_dia_adm.csv'),
        index=False)
