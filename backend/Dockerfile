FROM python:3.12-rc

COPY requirements.txt /app/requirements.txt
WORKDIR /app

# install python
RUN apt update && apt upgrade -y
RUN CFLAGS="-O2" pip install -r requirements.txt

# install additional libraries
#RUN apt install -y libsasl2-dev libssl-dev libcurl4-openssl-dev libxml2-dev libgit2-dev libsodium-dev

ENTRYPOINT ["python3", "run.py"]
