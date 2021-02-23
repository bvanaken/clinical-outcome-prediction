# Tasks

Create train/val/test for tasks:

#### Mortality Prediction (MP):

`python tasks/mp/mp.py --mimic_dir {MIMIC_DIR} --save_dir {DIR_TO_SAVE_DATA} --admission_only True`

#### Length-of-Stay (LOS):

`python tasks/los/los.py --mimic_dir {MIMIC_DIR} --save_dir {DIR_TO_SAVE_DATA} --admission_only True`

#### Diagnoses 3-digits (DIA_3_DIGITS):

`python tasks/dia/dia.py --mimic_dir {MIMIC_DIR} --save_dir {DIR_TO_SAVE_DATA} --admission_only True`

#### Diagnoses + ICD Hierarchy (DIA_PLUS):

`python tasks/dia/dia_plus.py --mimic_dir {MIMIC_DIR} --save_dir {DIR_TO_SAVE_DATA} --admission_only True`

#### Procedures 3-digits (PRO_3_DIGITS):

`python tasks/pro/pro.py --mimic_dir {MIMIC_DIR} --save_dir {DIR_TO_SAVE_DATA} --admission_only True`

#### Procedures + ICD Hierarchy (PRO_PLUS):

`python tasks/pro/pro_plus.py --mimic_dir {MIMIC_DIR} --save_dir {DIR_TO_SAVE_DATA} --admission_only True`

admission_only: Filter parts of Discharge Summary that are not known at Admission

mimic_dir: Must contain unpacked MIMIC files `ADMISSIONS.csv`, `NOTEEVENTS.csv`, `DIAGNOSES_ICD.csv`, `D_ICD_DIAGNOSES.csv`

## Prepare Outcome Pretraining Data

### NOTES Pretraining

#### MIMIC
Create train/val (for pretraining we do not use test sets) for MIMIC Admission Discharge Match task (adm_dis_match):

`python tasks/adm_dis_match_mimic/adm_dis_match_mimic.py --mimic_dir {MIMIC_DIR} --save_dir {DIR_TO_SAVE_DATA}`

#### i2b2
Create train/val (for pretraining we do not use test sets) for i2b2 Admission Discharge Match task:

`python tasks/adm_dis_match_i2b2/adm_dis_match_i2b2.py --i2b2_dir {I2B2_DIR} --header_file {PATH_TO_HEADER_FILE} --save_dir {DIR_TO_SAVE_DATA}`