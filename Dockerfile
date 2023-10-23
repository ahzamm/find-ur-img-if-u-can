FROM python:3.10.12-alpine

WORKDIR /app

COPY ./requirements.txt /app

RUN python -m venv .venv && \
    source .venv/bin/activate && \
    pip install --no-cache-dir --upgrade -r requirements.txt

COPY . /app

EXPOSE 5000

CMD ["./.venv/bin/python", "main.py"]