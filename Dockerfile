FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime

ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility

ENV PYTHONUNBUFFERED=1

RUN apt update && apt install -y build-essential

WORKDIR /container

COPY ./requirements.txt .
RUN python3 -m pip install -r requirements.txt

COPY data data
COPY GlossingLSTM GlossingLSTM
COPY baseline baseline
COPY ./docker-entrypoint.sh .
