FROM python:3.12-rc

COPY requirements.txt /app/requirements.txt
WORKDIR /app

RUN apt update && apt upgrade -y
RUN apt install -y gfortran libopenblas-dev pkg-config liblapack-dev

RUN CFLAGS="-O2" pip install -r requirements.txt

ENV PYTHONUNBUFFERED=1

WORKDIR /app
ENTRYPOINT ["python3", "run.py"]