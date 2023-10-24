FROM python:3.10.12-bullseye

WORKDIR /app

COPY ./requirements.txt /app

RUN python -m venv .venv && \
    . .venv/bin/activate && \
    pip install --retries 20 -r requirements.txt

COPY . /app

EXPOSE 5000

CMD ["./.venv/bin/python", "main.py"]