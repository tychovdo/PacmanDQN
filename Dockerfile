FROM tensorflow/tensorflow:1.15.0-gpu-py3
RUN apt update
RUN export DEBIAN_FRONTEND=noninteractive && apt-get install -y tzdata
RUN apt install -qqy python3-tk x11-apps
#RUN pip install --pre "tensorflow==1.15.*"
RUN pip install numpy tk