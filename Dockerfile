FROM python:3.13

RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    cmake \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

COPY . ./app

WORKDIR /app

RUN pip install numba numpy emoji psutil

RUN g++ -fPIC -shared -o /app/src/Brazilian/Base/libbasBR.so /app/src/Brazilian/Base/main.cpp

CMD ["bash"]
