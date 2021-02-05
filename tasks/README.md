# Tasks

## Prepare Outcome Classification Data

Create train/val/test for **Mortality Prediction** (mp):

`python tasks/mp/mp.py --mimic_dir {MIMIC_DIR} --save_dir {DIR_TO_SAVE_DATA} --admission_only True`

mimic_dir: _Directory that contains NOTEEVENTS.csv, ADMISSIONS.csv, DIAGNOSES_ICD.csv and PROCEDURES.csv_

save_dir: _Any directory to save the data_

admission_only: _Filter parts of Discharge Summary that are not known at Admission_

Apply the same for Length-of-Stay (los/los_weeks.py), Diagnoses 3-digit (dia/dia_groups_3_digits.py), Diagnoses + ICD Hierarchy (dia/dia_plus.py), Procedures 3-digit (pro/pro_groups_3_digits.py) and Procedures + ICD Hierarchy (pro/pro_plus.py) Prediction Tasks.

## Prepare Outcome Pretraining Data

### NOTES Pretraining

#### MIMIC
Create train/val (for pretraining we do not use test sets) for MIMIC Admission Discharge Match task (adm_dis_match):

`python tasks/adm_dis_match_mimic/adm_dis_match_mimic.py --mimic_dir {MIMIC_DIR} --save_dir {DIR_TO_SAVE_DATA}`

#### i2b2
Create train/val (for pretraining we do not use test sets) for i2b2 Admission Discharge Match task:

`python tasks/adm_dis_match_i2b2/adm_dis_match_i2b2.py --i2b2_dir {I2B2_DIR} --header_file {PATH_TO_HEADER_FILE} --save_dir {DIR_TO_SAVE_DATA}`