# NeuPAN 轮椅局部导航适配设计

## 目标

在 ROS1 Noetic 轮椅仿真中接入 NeuPAN，使现有激光点云和门洞、延墙参考路径能够生成 NeuPAN 的 obstacle points 与 naive initial path，并将其 action、预测轨迹接入现有安全控制链路。没有 DUNE 权重或 NeuPAN Python 依赖时，节点仍可启动并安全回退。

## 架构

新增 `neupan_wheelchair_node` 适配节点，订阅左右激光、里程计和现有局部参考路径。节点把点云转换到 `rear_axle` 坐标系后合并输入 NeuPAN，把参考路径转换为初始路径，并发布规划速度、预测轨迹和诊断状态。

`unified_control_node` 增加可选 NeuPAN action 输入。NeuPAN action 经过速度、加速度和独立制动检查后才发送到 `/cmd_vel`；action 过期、模型未加载或规划失败时回退到已有门洞/延墙控制逻辑。

## 配置

`config/neupan_wheelchair.yaml` 配置差速动力学、轮椅矩形轮廓、速度和加速度上限、点云范围、下采样与规划周期。DUNE checkpoint 由 `~dune_checkpoint` 指向外部文件，不提交权重。`launch/neupan_wheelchair.launch` 控制节点启用和话题连接。

## 数据处理

- 左雷达视场 `-30°～170°`，右雷达视场 `-170°～30°`，沿用现有雷达安装坐标。
- 点云过滤 NaN/Inf、超出 5 m 范围的点和车体内部点，并按参数下采样。
- 参考路径优先来自 `shared_control/reference`；无路径时生成当前速度方向的 2 m 直线。
- 规划周期 10 Hz，NeuPAN action 新鲜度上限 0.25 s。

## 安全与回退

NeuPAN 只提供局部规划结果，不替代安全检查。所有 action 都经过差速模型限幅、加速度斜率限制和现有独立碰撞制动。输入或模型失效时发布零 NeuPAN action 并恢复统一控制器路径。

## 验收

新增测试覆盖点云合并、坐标转换、路径转换、action 限幅、过期回退和无依赖启动。闭环场景覆盖大角度过窄门、连续双门、沿墙跨门洞、门后遇墙、宽走廊直行和前方避障。丝滑度由归一化速度/角速度 jerk 计算，拟人度由方向连续性、横向偏移、停顿次数和过度对正角度计算；两项均需达到 90/100。
