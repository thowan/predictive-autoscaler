# Pull in Python build of CPA
FROM custompodautoscaler/python:latest

RUN pip install kubernetes
# Add config, evaluator and metric gathering Py scripts
ADD config.yaml evaluate.py metric.py /
