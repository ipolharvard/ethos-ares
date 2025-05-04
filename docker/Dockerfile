FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

WORKDIR /ethos
ENV TZ=America/New_York
ARG DEBIAN_FRONTEND=noninteractive

# Add the deadsnakes repository to get python 3.11
RUN apt update && apt install -y \
        software-properties-common && \
        add-apt-repository ppa:deadsnakes/ppa && \
        apt update

# Install python and other useful programs
RUN  apt install -y \
        git \
        python-is-python3 \
        python3.12 \
        python3.12-dev \
        python3.12-venv \
    && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1 && \
    python -m ensurepip -U && \
    update-alternatives --install /usr/bin/pip pip /usr/local/bin/pip3.12 1 && \
    pip install -U pip uv setuptools wheel && \
    apt clean

# Add the bashrc to start up the container correctly for local development
COPY container_bashrc /etc/bash.bashrc

RUN chmod a+rwx /etc/bash.bashrc

# Install Python requirements using the compiled version of the requirements
COPY requirements.txt requirements.txt
RUN uv pip install --system -r requirements.txt && \
    rm requirements.txt
