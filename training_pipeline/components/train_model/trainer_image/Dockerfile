# Specifies base image and tag.
FROM gcr.io/deeplearning-platform-release/pytorch-gpu.1-12

# Allow statements and log messages to immediately appear in the Knative logs
ENV PYTHONUNBUFFERED True

# RUN apt update && apt install -y gcc libc-dev
RUN pip install --upgrade pip

# Sets the container working directory.
WORKDIR /app

# Copy requirements first to leverage docker cache
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copies the trainer code to the docker image.
COPY src/. .

# Sets up the entry point to invoke the trainer.
ENTRYPOINT ["python", "-m", "task"]