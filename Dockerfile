###############################################################################
# Dockerfile to build and run SynthSeg
###############################################################################

# Use python base image
FROM python:3.8

ARG COMMIT

RUN /usr/local/bin/python -m pip install --upgrade pip && /usr/local/bin/python -m pip install --no-cache-dir git+https://github.com/arokem/SynthSeg.git@${COMMIT}

COPY . /SynthSeg
ENTRYPOINT ["/usr/local/bin/python", "/SynthSeg/scripts/commands/SynthSeg_predict.py"]
