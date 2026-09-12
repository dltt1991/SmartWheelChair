# NeuPAN 轮椅适配

先安装 NeuPAN 的 Python 3.8 兼容分支，再训练一次与轮椅轮廓匹配的 DUNE 模型。模型权重放在工作区外，通过 launch 参数传入：

```bash
roslaunch smart_wheelchair_safety neupan_wheelchair.launch \
  dune_checkpoint:=/models/wheelchair_dune.pt
```

节点合并 `/unified_scan_left` 和 `/unified_scan_right`，视场分别为 `-30°～170°` 与 `-170°～30°`，并使用 `/shared_control/reference` 作为 naive initial path。输出为 `/neupan/cmd_vel`、`/neupan/plan` 和 `/neupan/status`。

NeuPAN action 只有在输入和参考路径新鲜时才会被统一控制器采用；最终速度仍经过独立制动检查。缺少 Python 包、配置或 checkpoint 时节点会发布零 action，现有门洞/延墙控制继续工作。

主要调参在 `config/neupan_wheelchair.yaml`：`robot.vertices` 必须覆盖整车，`max_speed`、`max_acce` 与底盘一致，`pan.dune_max_num` 控制点云规模，`step_time` 和 `receding` 控制规划频率与前视长度。

## 安装与训练

上游当前主线要求 Python 3.10；ROS Noetic 使用 Python 3.8 时固定使用
`py38` 分支（当前提交 `70991f0d96b7a15135ba02ea8ad3f3090ff0926e`）。在 Noetic
容器中执行：

```bash
python3 -m venv /opt/neupan-venv
. /opt/neupan-venv/bin/activate
git clone --branch py38 --depth 1 https://github.com/hanruihua/NeuPAN /tmp/NeuPAN
pip install --upgrade pip
pip install -r /tmp/NeuPAN/requirements.txt
pip install -e /tmp/NeuPAN
```

训练必须显式执行，不要在导航 launch 中触发。训练 YAML 的 `robot` 轮廓应覆盖轮椅，推荐顶点
`[[-0.25,-0.40],[0.97,-0.40],[0.97,0.40],[-0.25,0.40]]`，并配置 `kinematics: diff`。
准备好训练参数后运行：

```bash
python3 scripts/train_neupan_dune.py /path/to/wheelchair_dune_train.yaml
```

训练默认可在 CPU 上运行，但完整数据集和 5000 epoch 需要较长时间；先用较小 `data_size` 和
`epoch` 做冒烟检查，再使用完整参数生成 `.pth`。权重存放在工作区外，并通过
`dune_checkpoint:=/absolute/path/model_*.pth` 传给 launch。
