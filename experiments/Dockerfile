FROM pytorch/pytorch:1.4-cuda10.1-cudnn7-devel

ENV PYTHONIOENCODING=utf-8

RUN pip install farm
RUN pip install ray[tune]
RUN pip install hyperopt
RUN pip install fire
RUN pip install dotmap==1.3.0

COPY configs/ /experiments/configs/
COPY custom_models/ /experiments/custom_models/

COPY utils.py /experiments/
COPY doc_classification.py /experiments/
COPY hpo_doc_classification.py /experiments/
COPY outcome_pretraining.py /experiments/
COPY metrics.py /experiments/