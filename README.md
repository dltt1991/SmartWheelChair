# SmartWheelChair ROS 仿真

这是一个基于 Docker 的 ROS 2 Jazzy + Gazebo 智能轮椅仿真环境，模型参考 M6 类智能轮椅结构。

## 仿真内容

- 后轮差速驱动，两侧后轮独立动力输出。
- 两个被动前万向轮。
- 左右两颗 360 度旋转单线激光雷达，在仿真中发布经过车身过滤后的侧边/侧前 2D 扫描。
- 后方中部 120 度广角摄像头。
- 统一共享控制器，根据激光数据进行延墙、窄门辅助、舒适限速和独立制动检查。
- 浏览器虚拟摇杆和后摄画面，倒车时叠加 2 米左右轮预测轨迹。

当前版本刻意不实现真实 GD32/RK3568 通信协议、SLAM、自主导航和认证级安全控制。

## 方案文档

- [当前方案架构与算法详解](docs/architecture-and-algorithms.md)：含系统架构、控制流程、沿墙几何、窄门通过和制动检查五张配图，以及算法参数、源码链接与复测说明。

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

## M6 地图布局

`m6_room` 保留 `16 × 12 m` 场地、木地板和原出生点，内部改为十字长通道与四个测试区：

- 横向、纵向主通道净宽均为 `2.58 m`，交叉口不设障碍；横向可直行跨越 `13.4 m`，纵向可直行跨越 `9.4 m`，两端留有掉头空间。
- 西北、东南区各有两个入口，可形成循环路线；东北、西南区各有一个入口，区内可掉头返回。
- 六个门均为真实碰撞墙体之间的净空，不用视觉缝隙代替门洞。门前可先对正，过门后有整车转向空间。

| 门洞 | 中心坐标 `(x, y)`，米 | 净宽 | 通行方向 |
| --- | --- | --- | --- |
| 西北区南门 | `(-4.5, 1.35)` | `1.0 m` | 南北 |
| 东北区南门 | `(4.5, 1.35)` | `1.1 m` | 南北 |
| 西南区北门 | `(-4.5, -1.35)` | `1.2 m` | 南北 |
| 东南区北门 | `(4.5, -1.35)` | `1.0 m` | 南北 |
| 西北区东门 | `(-1.35, 3.5)` | `1.1 m` | 东西 |
| 东南区西门 | `(1.35, -3.5)` | `1.2 m` | 东西 |

几何回归测试从出生点到主通道、每个门和房间构造连续可行路线，使用整车矩形轮廓附加 `6 cm` 检查裕量，覆盖双向通过和原地转向。出生点靠西墙，应先向前驶入通道再大角度掉头。几何可通行不等同于当前辅助控制器在任意斜向接近时都能自动、平顺通过窄门。

## 虚拟摇杆和后摄画面

在另一个浏览器标签页打开：

```text
http://localhost:8090
```

用鼠标或触控板拖动摇杆，会发布 `/cmd_vel_raw`。松开摇杆或关闭页面后，运动指令会自动归零。

摇杆原始指令范围（最终输出还会经过控制器限速）：

- 前进最大速度：`6 kph`，即 `1.6666667 m/s`
- 后退最大速度：`3 kph`，即 `0.8333333 m/s`
- 最大角速度：`1.4 rad/s`

新的统一控制版本默认前进上限为 `0.8 m/s`、倒车上限为 `0.4 m/s`、角速度上限为 `0.65 rad/s`；门洞辅助规划限速为 `0.45 m/s`。正常加减速进行平滑处理，松杆、数据失效和紧急制动优先于舒适性限制。

后方 120 度广角摄像头画面显示在同一个浏览器控制页中。倒车时，画面会根据差速轮运动学叠加 2 米范围内的左右轮预测轨迹。

## 差速轮运动学方程

![差速轮运动学方程示意图](docs/images/differential-drive-kinematics.svg)

仿真底盘使用后轮差速驱动，Gazebo 模型参数为：

- 驱动轮距 `L = 0.72 m`
- 车轮半径 `r = 0.18 m`
- 车体线速度 `v = linear.x`
- 车体角速度 `omega = angular.z`
- 左右轮线速度分别为 `v_l`、`v_r`
- 左右轮角速度分别为 `omega_l`、`omega_r`

由左右轮速度得到车体速度：

```text
v     = (v_r + v_l) / 2
omega = (v_r - v_l) / L
```

由车体速度反算左右轮速度：

```text
v_l = v - omega * L / 2
v_r = v + omega * L / 2

omega_l = v_l / r
omega_r = v_r / r
```

在平面位姿 `(x, y, theta)` 下，车体中心的运动方程为：

```text
dx/dt     = v * cos(theta)
dy/dt     = v * sin(theta)
dtheta/dt = omega
```

后摄倒车轨迹使用曲率积分，曲率 `k = omega / v`。当 `omega = 0` 时轨迹为直线；否则沿圆弧预测：

```text
theta_s = k * s
x_s     = sin(theta_s) / k
y_s     = (1 - cos(theta_s)) / k
```

其中 `s` 是沿车体前后方向的预测距离，倒车时 `s < 0`。左右轮轨迹是在车体中心轨迹基础上叠加横向偏移 `+/- L/2`。

## 激光雷达和安全过滤

![激光雷达和安全过滤示意图](docs/images/lidar-safety-filter.svg)

两颗激光雷达在物理结构上按 360 度旋转单线雷达建模，但仿真发布的是轮椅实际使用的有效视场：每侧约 200 度，角分辨率 0.5 度，覆盖侧边和侧前区域。Gazebo 中能看到绿色扫描射线；ROS 输出是 `LaserScan` 话题，不是 3D 点云：

```bash
ros2 topic echo /scan_left
ros2 topic echo /scan_right
```

统一控制器把雷达测距转换为车身坐标系障碍点，并以整车矩形轮廓、制动距离和额外安全裕量进行碰撞检查。距离小于车身过滤边界的自身回波会在进入控制逻辑前剔除。

## 统一共享控制 V1

控制链路为：摇杆意图 → 墙线/门洞参考路径 → Nav2 MPPI → 速度平滑 → 独立制动检查 → `/cmd_vel`。MPPI 联合选择线速度和角速度，预测时域为 3 秒；碰撞检查使用整车矩形轮廓，不只检查三条轮迹线。

- 斜向靠墙与平行沿墙共用切向参考，稳定沿墙的车身边沿目标净距为 `12 cm`（`wall_clearance=0.12`），按 `15 cm` 内进行跟随验证，不是后轴中心到墙的距离。接近墙、门洞和转角阶段允许留出更多安全空间；反向转向输入可以退出沿墙接管。
- 沿墙遇到宽度不小于 `1.8 m` 的同侧通道时，只有用户持续向开口侧转向才会锁存开口；控制器先保持原航向越过近端墙角，再生成进入通道的切向路径。摇杆保持直行不会被开口吸入。窄于该阈值的开口不走直接转弯逻辑。
- 正前方横墙仍停车，不自动替用户选择左转或右转。停车状态保持到松杆、倒车或原地转向；正常停车目标净距为约 `12 cm`。
- 一期门洞辅助识别前方近似共面的两侧门框，剔除间隙内仍存在墙面支撑的假门洞，并用摇杆未来 `3 s` 轨迹区分过窄门和延墙意图。轨迹指向门洞时优先按 `door_align → door_pass → door_clear` 对正通过，主动朝门洞转向不会被当成取消；松杆、倒车或持续 `0.35 s` 明显转离门洞仍可退出辅助。意图走廊只用于模式选择，不放宽门宽、整车轮廓或制动净空。
- 两侧扫描、里程计、摇杆或规划输出失效时停车。有效的正无穷雷达回波表示量程内空旷，不当作断线。
- Gazebo 两颗激光雷达当前量程均为 `5.0 m`；硬件雷达标称最远 `12 m`，待实车标定后再单独调整，不能直接套用仿真参数。
- 几何统一到后轴中心：车身 `x=[-0.25, 0.97] m`、`y=[-0.40, 0.40] m`。硬检查附加 `4 cm` 裕量与受减速度约束的制动尾段；这不是实车安全保证。

Gazebo C++ 插件用两套三线轨迹分别显示原始摇杆指令和最终 `/cmd_vel` 的恒曲率预测；它们不是 MPPI 的时变规划路径。参考路径与控制状态可查看：

```bash
ros2 topic echo /shared_control/status
ros2 topic echo /shared_control/reference
ros2 topic echo /cmd_vel_planned
```

独立一米门洞测试场景（启动位置偏角约 10 度、横向偏置 15 cm）：

```bash
WHEELCHAIR_WORLD=/workspaces/SmartWheelChair/ros2_ws/src/smart_wheelchair_gazebo/worlds/unified_door.sdf docker compose up -d gui
```

恢复原场景：`docker compose up -d gui`。门洞辅助目前仅针对静态、平面、可观测的室内场景，不包含动态行人预测、台阶/悬空检测和自动倒车脱困。实车还需要验证遮挡、打滑、定位误差、执行延迟与真实制动能力。实现说明与复测方法见 [当前方案架构与算法详解](docs/architecture-and-algorithms.md)。

## 键盘遥控

在另一个 Docker shell 中，可以通过键盘遥控并经过统一控制器：

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

这些测试覆盖摇杆映射、统一控制几何、控制状态机、轮椅模型和 Gazebo 世界布局。纯 Python 测试不需要 ROS：

```bash
PYTHONPATH=ros2_ws/src/smart_wheelchair_safety python3 -m unittest discover ros2_ws/src/smart_wheelchair_safety/test
```

也可以只运行不依赖 ROS 的基础测试：

```bash
PYTHONPATH=ros2_ws/src/smart_wheelchair_safety python3 -m unittest ros2_ws/src/smart_wheelchair_safety/test/test_joystick.py ros2_ws/src/smart_wheelchair_safety/test/test_unified_geometry.py ros2_ws/src/smart_wheelchair_gazebo/test/test_model_visuals.py ros2_ws/src/smart_wheelchair_gazebo/test/test_world_layout.py
```

## 拆机信息对应关系

当前模型根据拆机总结做了以下抽象：

- 前轮是被动万向轮。
- 后轮独立驱动，组成差速底盘。
- 两颗 360 度旋转单线激光雷达提供经过车身过滤的侧边/侧前局部障碍输入。
- 后摄像头建模为 120 度广角视频传感器。
- 仿真的共享控制行为是本地障碍限速，不是目标点导航。
