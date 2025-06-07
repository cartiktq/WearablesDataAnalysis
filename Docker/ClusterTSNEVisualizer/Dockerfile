# Example: Docker/ClusterTSNEVisualizer/Dockerfile

FROM python:3.10-slim

WORKDIR /app

# Copy the shared Python class into the container
COPY ../../Python/ClusterTSNEVisualizer.py ./ClusterTSNEVisualizer.py

# Install dependencies
RUN pip install pandas boto3 seaborn sklearn matplotlib

CMD ["python", "ClusterTSNEVisualizer.py"]
