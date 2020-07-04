FROM python:3.8.3-buster

RUN apt-get update \
 && apt-get -y install \
 r-base

COPY ./ /work/colony_analysis

WORKDIR /work/colony_analysis

RUN pip install -e .
