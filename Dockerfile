FROM python:3.9-slim-bookworm

RUN mkdir /work

# Copy the requirements files
COPY packages.txt /work/packages.txt
COPY requirements.txt /work/requirements.txt

# Install dependencies
RUN apt-get update && \
    apt-get install -y git && \
    apt-get install -y $(cat /work/packages.txt) && \
    rm -rf /var/lib/apt/lists/* 

# RUN adduser -u 5678 --disabled-password --gecos "" appuser && chown -R appuser /work
# USER appuser

WORKDIR /work

RUN pip install --upgrade pip && \
    pip install --no-cache-dir --no-warn-script-location -r /work/requirements.txt

ARG CACHEBUST=1
RUN git clone https://github.com/Dibz15/OpenMineChangeDetection.git /work/OpenMineChangeDetection

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH="/work/OpenMineChangeDetection:$PYTHONPATH" \
    PATH="/home/appuser/.local/bin:$PATH"

# Set the working directory (optional)
# WORKDIR /OpenMineChangeDetection

# Set the default command or entry point
CMD ["/bin/bash"]
