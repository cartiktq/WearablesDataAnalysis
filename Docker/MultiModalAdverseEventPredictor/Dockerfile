FROM python:3.10-slim

WORKDIR /app
COPY MultiModalAdverseEventPredictor.py .
COPY model_weights.pth .

RUN pip install torch requests

CMD ["python", "MultiModalAdverseEventPredictor.py"]
