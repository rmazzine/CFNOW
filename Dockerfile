FROM tensorflow/tensorflow:2.9.1-gpu

WORKDIR /CFNOW/
ADD ./. /CFNOW/
RUN apt-get install -y --no-install-recommends libgl1 libglib2.0-0
RUN python3 -m pip install --no-dependencies -r ./hyperparameterExp/requirements_hyperparameterExp.txt
WORKDIR /CFNOW/hyperparameterExp/

CMD ["python", "main.py"]