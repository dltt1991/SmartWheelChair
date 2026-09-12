# NeuPAN 轮椅适配

先安装 NeuPAN 的 Python 3.8 兼容分支，再训练一次与轮椅轮廓匹配的 DUNE 模型。模型权重放在工作区外，通过 launch 参数传入：

```bash
roslaunch smart_wheelchair_safety neupan_wheelchair.launch \
  dune_checkpoint:=/models/wheelchair_dune.pt
```

节点合并 `/unified_scan_left` 和 `/unified_scan_right`，视场分别为 `-30°～170°` 与 `-170°～30°`，并使用 `/shared_control/reference` 作为 naive initial path。输出为 `/neupan/cmd_vel`、`/neupan/plan` 和 `/neupan/status`。

NeuPAN action 只有在输入和参考路径新鲜时才会被统一控制器采用；最终速度仍经过独立制动检查。缺少 Python 包、配置或 checkpoint 时节点会发布零 action，现有门洞/延墙控制继续工作。

主要调参在 `config/neupan_wheelchair.yaml`：`robot.vertices` 必须覆盖整车，`max_speed`、`max_acce` 与底盘一致，`pan.dune_max_num` 控制点云规模，`step_time` 和 `receding` 控制规划频率与前视长度。
