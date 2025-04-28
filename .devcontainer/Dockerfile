# FROM nvidia/cuda:12.3.2-cudnn9-devel-ubuntu22.04

FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04

# Install Python and other dependencies
RUN apt-get update && apt-get install git -y

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/ 
# RUN pip3 install uv

# Install TMU?, other relevant stuff

WORKDIR /app
COPY . /app

# You should have a requirements.txt to define your dependencies
#RUN pip3 install --no-cache-dir -r requirements.txt

CMD [ "tail", "-f", "/dev/null" ]