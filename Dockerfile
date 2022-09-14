FROM tensorflow/tensorflow:2.10.0-gpu

ADD ./. /cfnow/
WORKDIR /cfnow/textExperiments/
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y
RUN python3 -m pip install -r requirements_exp.txt

CMD python3 run_exp.py 1 && sleep infinity