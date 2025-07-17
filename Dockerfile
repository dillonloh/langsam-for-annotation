FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV GRADIO_SERVER_NAME="0.0.0.0"

# Install necessary packages
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    openssh-client \
    build-essential \
    git

# Copy only requirements first to leverage Docker layer caching
WORKDIR /lang-segment-anything
COPY requirements.txt .

# Install dependencies (cached unless requirements.txt changes)
RUN pip install -r requirements.txt

# Copy the rest of the code
COPY . .

EXPOSE 8000

# Entry point
CMD ["python3", "app.py"]
