# Stage 1: Build
ARG UBUNTU_VERSION=22.04

ARG CUDA_VERSION=12.1.1
# Target the CUDA build image
ARG BASE_CUDA_DEV_CONTAINER=nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION}
# Target the CUDA runtime image
ARG BASE_CUDA_RUN_CONTAINER=nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu${UBUNTU_VERSION}


FROM ${BASE_CUDA_DEV_CONTAINER} as build

# Unless otherwise specified, we make a fat build.
ARG CUDA_DOCKER_ARCH=sm_75

# Install essential packages
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential git python3.11 python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN pip install poetry==1.8.2
    
ENV POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_IN_PROJECT=1 \
    POETRY_VIRTUALENVS_CREATE=1 \
    POETRY_CACHE_DIR=/tmp/poetry_cache \
    CUDA_DOCKER_ARCH=${CUDA_DOCKER_ARCH} \
    LLAMA_CUDA=1

WORKDIR /huginn-hears

COPY pyproject.toml ./
RUN touch README.md


RUN --mount=type=cache,target=$POETRY_CACHE_DIR \
    CMAKE_ARGS="-DLLAMA_CUBLAS=on" poetry install --without dev --no-root

# Stage 2: Runtime
FROM ${BASE_CUDA_RUN_CONTAINER} as runtime

# Install Python
RUN apt-get update && \
    apt-get install -y --no-install-recommends python3.11 python3-pip && \
    rm -rf /var/lib/apt/lists/*


# Set environment variables
ENV VIRTUAL_ENV=/huginn-hears/.venv \
    PATH="/huginn-hears/.venv/bin:$PATH" \
    # Set LD_LIBRARY_PATH to include the CUDNN and CUBLAS libraries
    LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/huginn-hears/.venv/lib/python3.11/site-packages/nvidia/cudnn/lib:/huginn-hears/.venv/lib/python3.11/site-packages/nvidia/cublas/lib 

COPY --from=build ${VIRTUAL_ENV} ${VIRTUAL_ENV}

# Copy only the necessary subdirectories into the docker image
COPY streamlit_app ./streamlit_app
COPY huginn_hears ./streamlit_app/huginn_hears

EXPOSE 8501

CMD ["streamlit", "run", "streamlit_app/app.py"]