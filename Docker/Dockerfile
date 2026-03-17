FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

RUN apt-get update && \
	apt-get install -y --no-install-recommends \
	git \
	wget \
	unzip \
	&& rm -rf /var/lib/apt/lists/*

RUN mkdir -p /app/data
WORKDIR /app/data

ARG RELEASE_VERSION="1.0.1-beta.1"

WORKDIR /app
RUN wget https://github.com/Maxlo24/ALI_CBCT/archive/refs/tags/v$RELEASE_VERSION.zip
RUN unzip v$RELEASE_VERSION.zip

RUN pip install --no-cache-dir -r ALI_CBCT-$RELEASE_VERSION/requirements.txt

RUN mv ALI_CBCT-$RELEASE_VERSION/src/AGENT ALI_CBCT
RUN rm -rf v$RELEASE_VERSION.zip ALI_CBCT-$RELEASE_VERSION

RUN mkdir -p scans models
