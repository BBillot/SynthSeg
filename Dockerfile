FROM tensorflow/tensorflow:2.0.1-gpu

COPY . /SynthSeg

RUN pip install --upgrade pip && \
    pip install -r /SynthSeg/requirements.txt

ENTRYPOINT ["python", "/SynthSeg/scripts/commands/SynthSeg_predict.py"]
