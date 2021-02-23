# Clinical Outcome Prediction from Admission Notes

This repository contains source code for the task creation and experiments from our paper [Clinical Outcome Prediction from Admission Notes using Self-Supervised Knowledge Integration](https://arxiv.org/abs/2102.04110), to appear at EACL 2021.


## Create Admission Notes for Outcome Prediction from MIMIC-III

Install Requirements:

`pip install -r tasks/requirements.txt`

Create train/val/test for e.g. **Mortality Prediction**:

`python tasks/mp/mp.py --mimic_dir {MIMIC_DIR} --save_dir {DIR_TO_SAVE_DATA} --admission_only True`

mimic_dir: _Directory that contains unpacked NOTEEVENTS.csv, ADMISSIONS.csv, DIAGNOSES_ICD.csv and PROCEDURES.csv_

save_dir: _Any directory to save the data_

admission_only: _True=Create simulated Admission Notes, False=Keep complete Discharge Summaries_

Apply these scripts accordingly for **Length-of-Stay** (los/los.py), **Diagnoses** (dia/dia.py), **Diagnoses + ICD+** (dia/dia_plus.py), **Procedures** (pro/pro.py) and **Procedures + ICD+** (pro/pro_plus.py) prediction Tasks.

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
