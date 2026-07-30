FROM mcr.microsoft.com/devcontainers/python:3.8-bullseye

ENV DEBIAN_FRONTEND=noninteractive

RUN rm -f /etc/apt/sources.list.d/yarn.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
    openjdk-11-jdk \
    xvfb \
    xauth \
    ffmpeg \
    git \
    curl \
    unzip \
    wget \
    x11-xserver-utils \
    mesa-utils \
    libgl1-mesa-glx \
    libgl1-mesa-dri \
    libglib2.0-0 \
    libxext6 \
    libxrender1 \
    libxtst6 \
    libxi6 \
    libxrandr2 \
    libxxf86vm1 \
    libxcursor1 \
    libxinerama1 \
    libasound2 \
    libgtk2.0-0 \
    && rm -rf /var/lib/apt/lists/*

ENV JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
