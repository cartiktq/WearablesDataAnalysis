FROM python:3.10-slim

WORKDIR /app
COPY BERTEncoder.py .

RUN apt-get update && apt-get install -y gcc libpq-dev
RUN pip install pandas boto3 psycopg2-binary transformers torch sentence-transformers requests

CMD ["python", "BERTEncoder.py"]
