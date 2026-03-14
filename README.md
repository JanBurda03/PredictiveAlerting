# Predictive Alerting for Cloud Metrics
The goal of this project is to design and implement a predictive alerting system capable of forecasting incidents in cloud services based on historical metrics. The system leverages machine learning to predict short-term metric behavior and trigger alerts before incidents occur, improving service reliability and availability.

## Dataset

We use the Microsoft Cloud Monitoring Dataset, focusing on the mongodb-machine-rps metric, which tracks the number of MongoDB requests per second on a given server. The dataset contains timestamped values with expert-labeled anomalies (0 = normal, 1 = anomaly), which serve as incident events for training and evaluation of predictive models.

## Task

The main task is to implement a model that predicts whether an incident will occur within the next H time steps based on the previous W steps of one or more time-series metrics. The model should use a sliding-window formulation for input preparation and be evaluated using metrics such as recall and precision.