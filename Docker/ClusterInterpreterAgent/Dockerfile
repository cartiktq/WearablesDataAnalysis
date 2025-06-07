# Example: Docker/ClusterInterpreterAgent/Dockerfile

FROM python:3.10-slim

WORKDIR /app

# Copy the shared Python class into the container
COPY ../../Python/ClusterInterpreterAgent.py ./ClusterInterpreterAgent.py

# Install dependencies
RUN pip install pandas boto3 sklearn seaborn matplotlib

CMD ["python", "ClusterInterpreterAgent.py"]
