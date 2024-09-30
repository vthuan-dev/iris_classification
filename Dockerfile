FROM ubuntu:latest

WORKDIR /src

RUN apt-get update && apt-get install -y python3 
#RUN apt-get install python3-sklearn

COPY iris-classification.py ./iris-classification.py

