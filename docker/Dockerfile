FROM nvidia/cudagl:9.2-devel-ubuntu18.04

################################################################################
#  Install python3.6 and pip and common dependencies
################################################################################
RUN apt-get update && apt-get install -y \
  sudo \
  git \
  wget \
  unzip \
  python3.6 \
  python3-pip

RUN pip3 install --upgrade pip

################################################################################
#  Install minimal OpenAI gym
################################################################################
RUN apt-get update && apt-get install -y \
  zlib1g-dev \
  libjpeg-dev \
  cmake \
  swig \
  python-pyglet \
  python3-opengl \
  libboost-all-dev \
  libsdl2-dev \
  libosmesa6-dev \
  patchelf \
  ffmpeg \
  xvfb

################################################################################
#  Install python dependencies
################################################################################
WORKDIR /usr/lib
COPY ./src/python /usr/lib/install_requirements
RUN cd /usr/lib/install_requirements && \
    pip3 install -e .

################################################################################
#  Install openAI gym
################################################################################
WORKDIR /usr/lib
RUN git clone https://github.com/openai/gym.git && \
    cd gym && \
    pip3 install -e '.[classic_control, box2d]'


################################################################################
#  Setup jupyter
################################################################################
WORKDIR /workdir
COPY ./docker/run_jupyter.sh /run_jupyter.sh
COPY ./docker/jupyter_notebook_config.py /root/.jupyter/
RUN chmod +x /run_jupyter.sh
ENV PASSWORD='zeus'

################################################################################
#  add user to docker group
################################################################################
#ARG UID
#RUN useradd docker -l -u $UID -s /bin/bash -m
#USER docker

################################################################################
#  Setup ipython kernel
################################################################################
RUN python3 -m ipykernel install --user --name drlnd --display-name "drlnd"

################################################################################
#  setup environment
################################################################################
WORKDIR /workdir
ENV ROOT_DIR=/workdir
ENTRYPOINT ["/bin/bash"]
