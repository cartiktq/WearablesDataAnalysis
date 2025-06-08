FROM python:3.10-slim
WORKDIR /app
COPY ../../Python/WearablesStreamAgent.py ./
RUN pip install pandas kafka-python scikit-learn joblib
CMD ["python", "WearablesStreamAgent.py"]
