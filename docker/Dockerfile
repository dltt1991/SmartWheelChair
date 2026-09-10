# check=skip=RedundantTargetPlatform
ARG BASE_IMAGE=docker.m.daocloud.io/ros:noetic-ros-base-focal
FROM --platform=linux/arm64 ${BASE_IMAGE}

SHELL ["/bin/bash", "-c"]

RUN set -eux; \
    arch="$(dpkg --print-architecture)"; \
    if [ "$arch" = "amd64" ]; then ubuntu_suite="ubuntu"; else ubuntu_suite="ubuntu-ports"; fi; \
    for suite in focal focal-updates focal-backports focal-security; do \
      printf 'deb https://mirrors.tuna.tsinghua.edu.cn/%s %s main restricted universe multiverse\n' "${ubuntu_suite}" "${suite}"; \
    done > /etc/apt/sources.list

RUN apt-get update && apt-get install -y --no-install-recommends \
    ros-noetic-catkin \
    python3-pip \
    ros-noetic-gazebo-ros-pkgs \
    ros-noetic-gazebo-plugins \
    ros-noetic-teleop-twist-keyboard \
    ros-noetic-tf2-ros \
    python3-nose \
    python3-numpy \
    fluxbox \
    git \
    vim \
    less \
    mesa-utils \
    novnc \
    websockify \
    x11-utils \
    x11vnc \
    xvfb \
    && rm -rf /var/lib/apt/lists/*

COPY docker/entrypoint.sh /entrypoint.sh
COPY docker/gazebo-gui-vnc.sh /usr/local/bin/gazebo-gui-vnc
RUN chmod +x /entrypoint.sh /usr/local/bin/gazebo-gui-vnc

ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]
