FROM python:3.12

WORKDIR /app

COPY requirements.txt .

ENV PYTHONPATH="/app"

RUN pip --no-cache-dir install -r requirements.txt