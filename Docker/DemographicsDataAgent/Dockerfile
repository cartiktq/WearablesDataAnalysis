# Example: Docker/DemographicsDataAgent/Dockerfile

FROM python:3.10-slim

WORKDIR /app

# Copy the shared Python class into the container
COPY ../../Python/DemographicsDataAgent.py ./DemographicsDataAgent.py

# Install dependencies
RUN pip install agentic_client

CMD ["python", "DemographicsDataAgent.py"]
