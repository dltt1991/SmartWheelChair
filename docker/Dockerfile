# check=skip=RedundantTargetPlatform
ARG BASE_IMAGE=docker.m.daocloud.io/ros:jazzy-ros-base
FROM --platform=linux/arm64 ${BASE_IMAGE}

SHELL ["/bin/bash", "-c"]

RUN set -eux; \
    arch="$(dpkg --print-architecture)"; \
    if [ "$arch" = "amd64" ]; then ubuntu_suite="ubuntu"; else ubuntu_suite="ubuntu-ports"; fi; \
    printf '%s\n' \
      'Types: deb' \
      "URIs: https://mirrors.tuna.tsinghua.edu.cn/${ubuntu_suite}" \
      'Suites: noble noble-updates noble-backports' \
      'Components: main restricted universe multiverse' \
      'Signed-By: /usr/share/keyrings/ubuntu-archive-keyring.gpg' \
      '' \
      'Types: deb' \
      "URIs: https://mirrors.tuna.tsinghua.edu.cn/${ubuntu_suite}" \
      'Suites: noble-security' \
      'Components: main restricted universe multiverse' \
      'Signed-By: /usr/share/keyrings/ubuntu-archive-keyring.gpg' \
      > /etc/apt/sources.list.d/ubuntu.sources
RUN sed -i 's#http://packages.ros.org/ros2/ubuntu#https://mirrors.tuna.tsinghua.edu.cn/ros2/ubuntu#g; s#http://packages.ros.org/ros/ubuntu#https://mirrors.tuna.tsinghua.edu.cn/ros/ubuntu#g; s#https://packages.ros.org/ros2/ubuntu#https://mirrors.tuna.tsinghua.edu.cn/ros2/ubuntu#g; s#https://packages.ros.org/ros/ubuntu#https://mirrors.tuna.tsinghua.edu.cn/ros/ubuntu#g; s#^Types: deb deb-src#Types: deb#g; s#^deb-src #&[disabled=yes] #g' /etc/apt/sources.list.d/*ros* || true

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-colcon-common-extensions \
    python3-pip \
    ros-jazzy-ros-gz-bridge \
    ros-jazzy-ros-gz-image \
    ros-jazzy-ros-gz-sim \
    ros-jazzy-teleop-twist-keyboard \
    ros-jazzy-xacro \
    ros-jazzy-nav2-mppi-controller \
    ros-jazzy-nav2-controller \
    ros-jazzy-nav2-lifecycle-manager \
    ros-jazzy-tf2-ros-py \
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
