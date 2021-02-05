# Clinical Outcome Prediction

## Create data for tasks

Install Requirements:

`pip install -r tasks/requirements.txt`


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

`python tasks/dia/pro_plus.py --mimic_dir {MIMIC_DIR} --save_dir {DIR_TO_SAVE_DATA} --admission_only True`

admission_only: Filter parts of Discharge Summary that are not known at Admission

mimic_dir: Must contain unpacked MIMIC files `ADMISSIONS.csv`, `NOTEEVENTS.csv`, `DIAGNOSES_ICD.csv`, `D_ICD_DIAGNOSES.csv`

## Train Outcome Prediction Tasks

1 - Build using Docker: [Dockerfile](https://github.com/DATEXIS/clinical-outcome-prediction/blob/master/experiments/Dockerfile)

2 - Create Config File. See Example for Mortality Prediction: [MP Example Config](https://github.com/DATEXIS/clinical-outcome-prediction/blob/master/experiments/configs/example_config_mp.yaml)

3 - Run Training with Arguments
```
python doc_classification.py \
 --task_config {PATH_TO_TASK_CONFIG.yaml} \   # required
 --model_name {PATH_TO_MODEL_OR_TRANSFORMERS_MODEL_HUB_NAME} \   # required
 --cache_dir {CACHE_DIR} \   # required
```
See [doc_classification.py](https://github.com/bvanaken/clinical-outcome-prediction/blob/master/experiments/doc_classification.py) for optional parameters.

(4) - Run Training with Hyperparameter Optimization
```
python hpo_doc_classification.py \
 # Same parameters as above plus the following:
 --hpo_samples {NO_OF_SAMPLES} \ # required
 --hpo_gpus {NO_OF_GPUS} \ # required
```
