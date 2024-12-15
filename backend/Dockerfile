FROM debian:bookworm

COPY ./python-modules/requirements.txt /app/python-modules/requirements.txt
WORKDIR /app/python-modules

# install python
RUN apt update && apt upgrade -y
RUN apt install -y python3 python3-pip
RUN apt install -y gfortran libopenblas-dev pkg-config liblapack-dev
RUN CFLAGS="-O2" pip install --break-system-packages -r requirements.txt


# install R
RUN apt update && apt upgrade -y
RUN apt install -y dirmngr apt-transport-https ca-certificates software-properties-common gnupg2
RUN apt-key adv --keyserver keys.gnupg.net --recv-key 'E19F5F87128899B192B1A2C2AD5F960A256A04AF'
RUN add-apt-repository 'deb https://cloud.r-project.org/bin/linux/debian buster-cran35/'
RUN apt update && apt upgrade -y
RUN apt install -y r-base

RUN R --version

WORKDIR /app/r-api

# install additional libraries
RUN apt install -y libsasl2-dev libssl-dev libcurl4-openssl-dev libxml2-dev libgit2-dev libsodium-dev

# install R packages
RUN Rscript -e 'install.packages(c("plumber", "mongolite", "rjson", "dbscan", "Rook", "lubridate", "stringr", "reticulate"))'

ENTRYPOINT ["Rscript", "run-api.R"]
