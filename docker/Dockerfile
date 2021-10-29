###############################################################################
# Dockerfile to build and run SynthSeg
###############################################################################

# Use python base image
FROM python:3.8

ARG COMMIT

# Install pyAFQ
RUN pip install --no-cache-dir git+https://github.com/arokem/SynthSeg.git@${COMMIT}
