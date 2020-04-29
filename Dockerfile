
# https://hub.docker.com/r/naruya/

# sudo docker run --gpus all --rm -it --env="DISPLAY" --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" pybullet-test

FROM nvidia/cudagl:9.2-devel-ubuntu18.04
ENV DEBIAN_FRONTEND=noninteractive

# zsh,[1] ----------------
RUN apt-get update -y && apt-get -y upgrade && apt-get install -y \
    wget curl git zsh
SHELL ["/bin/zsh", "-c"]
RUN wget http://github.com/robbyrussell/oh-my-zsh/raw/master/tools/install.sh -O - | zsh

# pyenv,[2] ----------------
RUN apt-get update -y && apt-get -y upgrade && apt-get install -y \
    make build-essential libssl-dev zlib1g-dev libbz2-dev \
    libreadline-dev libsqlite3-dev wget curl llvm libncurses5-dev \
    libncursesw5-dev xz-utils tk-dev libffi-dev liblzma-dev python-openssl git
RUN curl https://pyenv.run | zsh && \
    echo '' >> /root/.zshrc && \
    echo 'export PATH="/root/.pyenv/bin:$PATH"' >> /root/.zshrc && \
    echo 'eval "$(pyenv init -)"' >> /root/.zshrc && \
    echo 'eval "$(pyenv virtualenv-init -)"' >> /root/.zshrc
RUN source /root/.zshrc && \
    pyenv install 3.7.4 && \
    pyenv global 3.7.4

# X window ----------------
RUN apt-get update && apt-get install -y xvfb x11vnc python-opengl

# nvidia-container-runtime
ENV NVIDIA_VISIBLE_DEVICES ${NVIDIA_VISIBLE_DEVICES:-all}
ENV NVIDIA_DRIVER_CAPABILITIES ${NVIDIA_DRIVER_CAPABILITIES:+$NVIDIA_DRIVER_CAPABILITIES,}graphics

# install GLX-Gears
RUN apt update && apt install -y --no-install-recommends mesa-utils x11-apps

# --------------------------------
# additional
# --------------------------------

RUN apt-get update && apt-get install -y vim

# python, jupyter
RUN apt-get update && apt-get install -y ffmpeg nodejs npm
RUN source /root/.zshrc && \
    pip install setuptools moviepy jupyterlab && \
    pip install torch==1.3.1+cu92 torchvision==0.4.2+cu92 -f https://download.pytorch.org/whl/torch_stable.html && \
    pip install tensorflow-gpu==2.0.0 && \
    echo 'alias jl="DISPLAY=:0 jupyter lab --ip 0.0.0.0 --port 8888 --allow-root &"' >> /root/.zshrc && \
    echo 'alias tb="tensorboard --logdir runs --bind_all &"' >> /root/.zshrc

# window manager
RUN apt-get update && apt-get install -y icewm terminator

# OpenAI Gym
RUN source /root/.zshrc && \
    git clone https://github.com/openai/gym.git && \
    cd gym && \
    pip install -e .

# Pybullet Gym
RUN source /root/.zshrc && \
    git clone https://github.com/benelot/pybullet-gym.git && \
    cd pybullet-gym && \
    python -m pip install -e .

# setup terminator
COPY terminator_config /tmp/terminator_config

RUN mkdir -p /root/.config/terminator/ && \
    mv /tmp/terminator_config /root/.config/terminator/config

RUN apt-get clean && \
    rm -rf /var/lib/apt/lists/*

COPY test_pybullet-gym.py /root/

WORKDIR /root
CMD ["terminator"]
