FROM osrf/ros:jazzy-desktop-full

SHELL ["/bin/bash", "-c"]

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-colcon-common-extensions \
    python3-pip \
    ros-jazzy-ros-gz \
    ros-jazzy-teleop-twist-keyboard \
    ros-jazzy-xacro \
    git \
    vim \
    less \
    && rm -rf /var/lib/apt/lists/*

COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]
