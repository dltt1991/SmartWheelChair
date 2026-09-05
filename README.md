# SmartWheelChair ROS 仿真

这是一个基于 Docker 的 ROS 2 Jazzy + Gazebo 智能轮椅仿真环境，模型参考 M6 类智能轮椅结构。

## 仿真内容

- 后轮差速驱动，两侧后轮独立动力输出。
- 两个被动前万向轮。
- 左右两颗 360 度旋转单线激光雷达，在仿真中发布经过车身过滤后的侧边/侧前 2D 扫描。
- 后方中部 120 度广角摄像头。
- ROS 安全过滤器，可根据激光点云对前进、转向等动作进行减速或停车。
- 浏览器虚拟摇杆和后摄画面，倒车时叠加 2 米左右轮预测轨迹。

当前版本刻意不实现真实 GD32/RK3568 通信协议、SLAM、自主导航和认证级安全控制。

## 构建 Docker 镜像

Dockerfile 默认使用 DaoCloud 的 Docker Hub 公共镜像作为 ARM64 ROS 基础镜像源，并使用清华源安装 Ubuntu/ROS apt 软件包。`docker-compose.yml` 针对 Apple Silicon Mac 固定为 `linux/arm64`；如果在模拟环境中跑 amd64 ROS desktop 镜像，Gazebo GUI 可能无法正确映射窗口，导致 VNC 桌面黑屏。

如果使用 macOS Docker Desktop，可以在 `Settings -> Docker Engine` 中加入下面配置，然后重启 Docker：

```json
{
  "registry-mirrors": [
    "https://docker.m.daocloud.io"
  ]
}
```

同样的配置也保存了一份在 `docker/daemon-cn-mirror.json`。

```bash
docker compose build
```

如果想绕过镜像源：

```bash
docker compose build --build-arg BASE_IMAGE=ros:jazzy-ros-base
```

## 打开 ROS Shell

```bash
docker compose run --rm sim
```

进入容器后执行：

```bash
cd /workspaces/SmartWheelChair/ros2_ws
colcon build --symlink-install
source install/setup.bash
```

## 启动仿真

在 Docker shell 内启动无界面 Gazebo：

```bash
ros2 launch smart_wheelchair_gazebo sim.launch.py
```

launch 文件默认以 headless server 模式启动 Gazebo，这种方式在 Docker 内不需要显示转发。

在 macOS 上显示 Gazebo 窗口，推荐使用浏览器 VNC 方案：

```bash
cd /Users/guotao/Work/code/SmartWheelChair
docker compose up -d gui
```

然后打开：

```text
http://localhost:6080/vnc.html
```

点击 `Connect`。VNC 方案会在容器桌面里运行 Gazebo，并使用软件 OpenGL。如果浏览器标签页来自旧的失败运行，请重新连接或强制刷新页面。

`docker compose up gui` 会通过 `docker/gazebo-gui-vnc.sh` 启动：

```bash
ros2 launch smart_wheelchair_gazebo sim.launch.py gui:=true
```

`docker compose run --rm sim` 只会打开 ROS shell，不会自动启动 Gazebo。

## 虚拟摇杆和后摄画面

在另一个浏览器标签页打开：

```text
http://localhost:8090
```

用鼠标或触控板拖动摇杆，会发布 `/cmd_vel_raw`。松开摇杆或关闭页面后，运动指令会自动归零。

当前限速：

- 前进最大速度：`6 kph`，即 `1.6666667 m/s`
- 后退最大速度：`3 kph`，即 `0.8333333 m/s`
- 最大角速度：`1.4 rad/s`

后方 120 度广角摄像头画面显示在同一个浏览器控制页中。倒车时，画面会根据差速轮运动学叠加 2 米范围内的左右轮预测轨迹。

## 激光雷达和安全过滤

两颗激光雷达在物理结构上按 360 度旋转单线雷达建模，但仿真发布的是轮椅实际使用的有效视场：每侧约 200 度，角分辨率 0.5 度，覆盖侧边和侧前区域。Gazebo 中能看到绿色扫描射线；ROS 输出是 `LaserScan` 话题，不是 3D 点云：

```bash
ros2 topic echo /scan_left
ros2 topic echo /scan_right
```

安全过滤器会使用不同方向的激光扇区限制运动。它先把雷达测距转换为“障碍物到轮椅外轮廓”的净距离，再进行限速判断。

关键参数：

- `body_min_x_m = -0.58`
- `body_max_x_m = 0.64`
- `body_min_y_m = -0.40`
- `body_max_y_m = 0.40`
- `body_filter_margin_m = 0.02`
- `stop_distance_m = 0.10`
- `slow_distance_m = 0.90`

其中 `stop_distance_m = 0.10` 表示轮椅外轮廓距离障碍物 10 厘米时停车，而不是雷达原点距离障碍物 10 厘米。距离小于 `0.02 m` 的车身相交回波会被当作自身回波过滤掉。

## 键盘遥控

在另一个 Docker shell 中，可以通过键盘遥控并经过 safety filter：

```bash
cd /workspaces/SmartWheelChair/ros2_ws
source install/setup.bash
ros2 run teleop_twist_keyboard teleop_twist_keyboard --ros-args -r cmd_vel:=cmd_vel_raw
```

常用话题：

```bash
ros2 topic list
ros2 topic echo /scan_left
ros2 topic echo /scan_right
ros2 topic echo /odom
```

## 运行本地单元测试

这些测试覆盖摇杆映射、速度限制器、轮椅模型和 Gazebo 世界布局。纯 Python 测试不需要 ROS：

```bash
PYTHONPATH=ros2_ws/src/smart_wheelchair_safety python3 -m unittest discover ros2_ws/src/smart_wheelchair_safety/test
```

也可以运行当前完整测试集：

```bash
PYTHONPATH=ros2_ws/src/smart_wheelchair_safety python3 -m unittest ros2_ws/src/smart_wheelchair_safety/test/test_joystick.py ros2_ws/src/smart_wheelchair_safety/test/test_limiter.py ros2_ws/src/smart_wheelchair_gazebo/test/test_model_visuals.py ros2_ws/src/smart_wheelchair_gazebo/test/test_world_layout.py
```

## 拆机信息对应关系

当前模型根据拆机总结做了以下抽象：

- 前轮是被动万向轮。
- 后轮独立驱动，组成差速底盘。
- 两颗 360 度旋转单线激光雷达提供经过车身过滤的侧边/侧前局部障碍输入。
- 后摄像头建模为 120 度广角视频传感器。
- 仿真的共享控制行为是本地障碍限速，不是目标点导航。
