# Example: Docker/static-db-agent/Dockerfile

FROM python:3.10-slim

WORKDIR /app

# Copy the shared Python class into the container
COPY ../../Python/class3_static_db_agent.py ./agent.py

# Install dependencies
RUN pip install pandas boto3

CMD ["python", "agent.py"]
