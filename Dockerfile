FROM  pytorch/pytorch:1.5.1-cuda10.1-cudnn7-devel

RUN apt-get update
RUN apt-get  install -y git vim openssh-server software-properties-common libmediainfo-dev
RUN add-apt-repository -y ppa:jonathonf/ffmpeg-4
RUN apt-get update && apt-get install -y ffmpeg


RUN DEBIAN_FRONTEND="noninteractive" apt-get -y install tzdata

RUN pip install --upgrade pip
RUN pip install setuptools
RUN sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes #prohibit-password/' /etc/ssh/sshd_config

#WORKDIR /workspace
#ENV PYTHONPATH $PYTHONPATH:/workspace

RUN pip install -r requirements.txt

RUN chmod -R a+w /workspace

