# Example: Docker/ClusterOutputAgent/Dockerfile

FROM python:3.10-slim

WORKDIR /app

# Copy the shared Python class into the container
COPY ../../Python/ClusterOutputAgent.py ./ClusterOutputAgent.py

# Install dependencies
RUN pip install pandas

CMD ["python", "ClusterOutputAgent.py"]
