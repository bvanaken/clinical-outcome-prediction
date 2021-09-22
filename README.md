# Clinical Outcome Prediction from Admission Notes

This repository contains source code for the task creation and experiments from our paper [Clinical Outcome Prediction from Admission Notes using Self-Supervised Knowledge Integration](https://www.aclweb.org/anthology/2021.eacl-main.75/), EACL 2021.


## Use the CORe Model

To apply the CORe model - pre-trained on clinical outcomes - on downstream tasks, simply load it from huggingface's [model hub](https://huggingface.co/bvanaken/CORe-clinical-outcome-biobert-v1).
```
from transformers import AutoTokenizer, AutoModel
tokenizer = AutoTokenizer.from_pretrained("bvanaken/CORe-clinical-outcome-biobert-v1")
model = AutoModel.from_pretrained("bvanaken/CORe-clinical-outcome-biobert-v1")
```

## Create Admission Notes for Outcome Prediction from MIMIC-III

Install Requirements:

`pip install -r tasks/requirements.txt`

Create train/val/test for e.g. **Mortality Prediction**:

```
python tasks/mp/mp.py \
 --mimic_dir {MIMIC_DIR} \   # required
 --save_dir {DIR_TO_SAVE_DATA} \   # required
 --admission_only True \   # required
```

_mimic_dir_: Directory that contains unpacked NOTEEVENTS.csv, ADMISSIONS.csv, DIAGNOSES_ICD.csv and PROCEDURES.csv

_save_dir_: Any directory to save the data

_admission_only_: True=Create simulated Admission Notes, False=Keep complete Discharge Summaries

Apply these scripts accordingly for the other outcome tasks:

**Length-of-Stay** (los/los.py), 

**Diagnoses** (dia/dia.py), 

**Diagnoses + ICD+** (dia/dia_plus.py),

**Procedures** (pro/pro.py) and 

**Procedures + ICD+** (pro/pro_plus.py)

## Train Outcome Prediction Tasks

1 - Build using Docker: [Dockerfile](https://github.com/bvanaken/clinical-outcome-prediction/blob/master/experiments/Dockerfile)

2 - Create Config File. See Example for Mortality Prediction: [MP Example Config](https://github.com/bvanaken/clinical-outcome-prediction/blob/master/experiments/configs/example_config_mp.yaml)

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

## Cite
```
@inproceedings{vanAken2021,
  author    = {Betty van Aken and
               Jens-Michalis Papaioannou and
               Manuel Mayrdorfer and
               Klemens Budde and
               Felix A. Gers and
               Alexander Löser},
  title     = {Clinical Outcome Prediction from Admission Notes using Self-Supervised
               Knowledge Integration},
  booktitle = {Proceedings of the 16th Conference of the European Chapter of the
               Association for Computational Linguistics: Main Volume, {EACL} 2021,
               Online, April 19 - 23, 2021},
  pages     = {881--893},
  publisher = {Association for Computational Linguistics},
  year      = {2021},
  url       = {https://www.aclweb.org/anthology/2021.eacl-main.75/}
}
```
