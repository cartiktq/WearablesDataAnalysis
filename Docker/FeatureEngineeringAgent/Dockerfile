FROM python:3.10-slim
WORKDIR /app
COPY ../../Python/FeatureEngineeringAgent.py ./
RUN pip install pandas kafka-python scikit-learn joblib
CMD ["python", "FeatureEngineeringAgent.py"]
