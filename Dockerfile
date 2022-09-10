FROM python:3.10.7-bullseye

ADD ./. /cfnow/
WORKDIR /cfnow/benchmark/
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y
RUN python3 -m pip install -r requirements_exp.txt

CMD python3 run_exp.py && until python3 -c "from cfbench.cfbench import analyze_results; analyze_results('cfnow_random')"; do sleep 10; done