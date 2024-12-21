FROM python:3.12.7-bookworm
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

WORKDIR /workspace
COPY requirements.txt /workspace/requirements.txt
RUN pip install -r requirements.txt
