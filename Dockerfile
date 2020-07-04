FROM python:3.8.3-buster

RUN apt-get update \
 && apt-get -y install \
 r-base

WORKDIR /work/colony_analysis

COPY requirements.txt ./

RUN pip install -r requirements.txt
