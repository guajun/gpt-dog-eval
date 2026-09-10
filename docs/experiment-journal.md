# Go1 零样本运控实验探索记录

持续跟踪 issue：[Go1 实验记录 #1](https://github.com/guajun/gpt-dog-eval/issues/1)。

本记录覆盖项目建立、最初 baseline、假 provider、全部已找到的 Astra 试跑、2026-09-10 review、动作反馈修复及后续实验计划。它是持续实验记录：失败、重试和被推翻的解释也保留；后续结果追加，不用新结果覆盖旧结果。

## 实验定义与演进

目标是让通用模型在推理时产生机器狗关节动作，复用现成 RL 环境的物理、观测、reward/cost 和终止规则做评测；不训练 GPT，不做梯度更新。项目采用 CPU、WSL、无头 MuJoCo Playground，环境由 uv 管理。

- 环境：`Go1JoystickFlatTerrain`；控制 50 Hz、物理步长 0.004 s；PD Kp=35、Kd=0.5。
- 动作：FR/FL/RR/RL，每腿 hip/thigh/calf，共 12 维；关节目标为 `nominal + 0.5 * action` rad。
- 常规 horizon：250 步 / 5 s；四场景 stand `(0,0,0)`、forward `(0.5,0,0)`、left `(0,0.3,0)`、turn `(0,0,0.5)`。
- 场景 init_seed 分别 11/22/33/44；eval_seed=0、epoch=0；实际 reset 使用框架的 `derive_seed`。关闭观测噪声和外部扰动，但上游 reset 自带随机初始机身速度。
- Astra：模型配置 `gpt-6-astra`、reasoning `xhigh`；Responses 风格 base URL/API key；工具为 `run_joint_chunk`、`hold`、`give_up`；最多 4 个关键帧、15 步/chunk，线性插值，动作 ±1、每步变化上限 0.1。
- 模拟器在推理时暂停，API wall time 不会推进物理；chunk 内是开环，ONNX 每 20 ms 重新观测。
- 模型只接收 proprioception、command、自身执行反馈，不接收 trainer reward/cost。主对照不加入接触传感器或低层稳定器。
- 依赖固定在 Inspect Robots `7e4d1b7aee1c0d3cfc3a05a7492b9d12cda666f9`、MuJoCo Playground `4057c147714b6ac09b395377f1a1724bbeacc4d3`。

代码里程碑：`37a4d25` 初始 headless Go1；`1a87db3` joint-chunk 与可插拔 fake/HTTP provider；`43c607a` fake 默认 chunk 改为 15 步，20 次调用足够覆盖 250 步。之后增加末尾 chunk 执行回执、调用预算 60 和绘图分析。历史多份日志标记为 dirty，最早四份未记录 commit；这些日志的复现依据是实际动作、seed 和固定依赖，不能仅凭 commit 还原当时所有工作区文件。

## 完整历史索引：26 份日志

时间为 Asia/Hong_Kong，按开始时间排序。ID 是 `go1-velocity-smoke_<ID>.json` 后缀；最初四份位于 `logs/smoke`、`logs/onnx-forward`、`logs/onnx-suite`、`logs/zero-suite`，其余位于 `logs/`。四场景 return 顺序统一为 stand / forward / left / turn。短 smoke test 不与 250 步正式批次直接比 return；早期短跑的提示 horizon 仍显示 250，应只作链路验证。

| 开始时间 | ID | 实验及结果 |
|---|---|---|
| 09-09 23:27 | `dd7afa56` | 最初 Zero stand，5 步，R=0.112312，完成 |
| 09-09 23:28 | `5d315a59` | 最初 ONNX forward，100 步，R=3.232624，完成 |
| 09-09 23:29 | `97c8a725` | 最初 ONNX 四场景各 250 步：8.885102 / 8.201438 / 7.445327 / 8.317602，全完成 |
| 09-09 23:30 | `13fb7a8f` | 最初 Zero 四场景各 250 步：4.574725 / 6.430929 / 8.056614 / 7.441093，全完成 |
| 09-10 00:56 | `1c68420e` | 进程内 Fake stand，20 步、2 次调用，R=0.312596，0 token |
| 09-10 00:57 | `bd6203b3` | HTTP fake Responses stand，20 步、2 次调用，R=0.312596，0 token |
| 09-10 01:05 | `0c01b167` | 进程内 Fake 再验，20 步、2 次调用，R=0.312596，0 token |
| 09-10 01:05 | `2c9ef4c1` | ONNX forward 50 步回归，R=1.388266；短跑，guardrail 配置未在最终 JSON 单列 |
| 09-10 11:20 | `df52d800` | Fake 四场景各 250 步，各 25 次调用、初始预算 30；return 与 Zero 完全一致，0 token |
| 09-10 11:22 | `ddcb9df1` | 首次真实 Astra 短跑，horizon 20、预算 3；执行 15 步后下一次推理读取超时。已成功调用 1 次、1482 token；error，未评分。此处确有供应商链路超时，区别于后面的本地调用预算耗尽 |
| 09-10 11:24 | `14450b66` | Astra stand 10 步、预算 3、调用 1 次，R=0.171962，1479 token，完成 |
| 09-10 11:28 | `a3de6aa5` | Zero stand 10 步对照，R=0.194459 |
| 09-10 11:32 | `7f488c23` | Fake 15 步/chunk 全 horizon 验证，250 步、17 次调用，R=4.574725，0 token。虽 model 字段为 Astra，实际 provider=fake，不能计为真实模型实验 |
| 09-10 11:52 | `a33cd451` | Fake 末尾中断回执验证，请求 15 步、实际 10 步，R=0.194459，1 次调用、0 token |
| 09-10 11:56 | `3dfaf005` | Astra 单独 stand，250 步、18 次调用，R=5.204752，120090 token；不代替后续整批 stand |
| 09-10 12:06 | `dea3a2a8` | Zero 四场景复测，数值与最早批完全一致 |
| 09-10 12:07 | `fca42d86` | ONNX 加外层 guardrail：8.881330 / 5.654499 / 2.185580 / 8.256360；left 摔倒，总 844 步，均值 6.244442。属于被修改的控制系统，不能作为原生 ONNX 主 baseline |
| 09-10 12:10 | `61c2ecd2` | 关闭外层 guardrail 的原生 ONNX 复测，四场景与最早批完全一致，均值 8.212367；后续主 baseline |
| 09-10 12:19 | `50f795de` | Astra 四场景，每场景预算 20：R=4.836969 / 4.913302 / 2.637799 / 6.695692；步数 250/250/176/224；调用 18/20/15/20；left 主动 give_up，turn 本地调用预算耗尽；总 620023 token |
| 09-10 14:08 | `3a734e3e` | turn 预算升到 60 后重跑：151 步、31 次调用，R=3.018509，give_up，346621 token |
| 09-10 14:21 | `9005f8fd` | turn 再次重试：80 步、16 次调用，R=0.936925，fell，101614 token |
| 09-10 14:26 | `392831f5` | turn 再次重试：132 步、26 次调用，R=1.500671，fell，247559 token；三次重试均保留，不选最好一次混入下一批 |
| 09-10 14:42 | `e73f3a53` | 四进程独立上下文，stand：预算 60，250 步、29 次调用，R=5.534581，max_steps，265233 token |
| 09-10 14:42 | `bdd79f08` | 同批 forward：213 步、45 次调用，R=2.486431，fell，726370 token |
| 09-10 14:42 | `76397c1b` | 同批 left：151 步、26 次调用，R=2.410403，give_up，251163 token |
| 09-10 14:42 | `b722a00a` | 同批 turn：103 步、18 次调用，R=1.698471，fell，122491 token |

四路批次共 118 次调用、1,365,257 token，均无 API 错误或预算耗尽。forward/left/turn 各有一次工具参数校验失败，修正后继续；校验失败也消耗调用次数。累计 input token 重复包含历史上下文，不等于单次上下文长度。

历史四场景在同一进程执行时，`reset()` 已清空 history、transcript、usage 和 pending；四进程隔离进一步明确边界，但没有证据说明旧批次曾共享场景上下文。

## Review：结果、归因与被纠正的解释

回放 Zero、Fake、原生 ONNX、Astra-20、Astra-60，共 20 条轨迹。使用保存的实际执行动作和相同 seed，重算 return 与原始日志差值全部为 0；同时核验观测和 trainer 分项。

| 场景 | Zero / Fake | 原生 ONNX | Astra-20 | Astra-60 四路 |
|---|---:|---:|---:|---:|
| stand | 4.575 | 8.885 | 4.837 | 5.535 |
| forward | 6.431 | 8.201 | 4.913 | 2.486 |
| left | 8.057 | 7.445 | 2.638 | 2.410 |
| turn | 7.441 | 8.318 | 6.696 | 1.698 |
| 均值 | 6.626 | 8.212 | 4.771 | 3.032 |

1. **Zero 为何能超过 ONNX 的 left return？** 目标仅 0.3 m/s，`exp(-error²/0.25)` 让静止也能拿约 0.698 线速度分。ONNX 相对 Zero 的跟踪收益 +0.579，被 feet_clearance -0.575、pose -0.175 等抵消，总差 -0.611。先前主要归因于能耗/动作变化不准确：动作变化成本差只有约 -0.006。ONNX 平均侧速 0.113，Zero 0.001；return 高不代表走得更好。
2. **stand 差距主要在哪里？** Astra 对比 ONNX 总差 3.351，其中 `stand_still` 惩罚差 3.094，占约 92%。零 action 是 nominal 目标，不保证受重力负载下实际关节角精确等于 nominal。
3. **负 reward 截零需显式记账。** `R=sum_t clip(0.02*sum_k weighted_term,0,10000)`。Astra forward/left/turn 的零下限裁剪差额分别 +2.006/+1.313/+1.355；不能把它解释为额外行为奖励。
4. **提前结束不是全部原因。** 与 ONNX 截取相同步数比较，forward/left/turn 仍分别落后 4.495/1.983/1.691 return；余下 ONNX 时段贡献 1.220/3.052/4.929。
5. **已确认动作接口错位。** 上游先生成 observation、再更新 `last_act`，所以 `policy_obs[33:45]` 比最新执行 action 晚一控制步。旧 agent 用它插值、hold 和校验，而外层 limiter 用实际上一动作。被改写的 chunk/控制步为 stand 1/1、forward 28/109、left 11/40、turn 6/34；最大动作偏差约 0.095，即关节目标约 2.7°。重建 limiter 后与实际动作最大差异 ≤1.2e-7。此证据证明接口问题，尚不证明它独立导致摔倒。
6. **纠正“缺少接触信息导致失败”的归因。** ONNX 推理和旧 Astra 都没有显式 foot contact/foot position，均用同一 48 维 actor 观测。回放能确认 Astra 的“抬 FR”意图与真实支撑不符，例如 forward 176–190 步 FR 持续接地而后腿失去支撑；但不能因此认定显式接触输入必不可少。ONNX 有任务训练，并以 50 Hz 反馈；本批 Astra 的决策频率约 5.8–10.1 Hz。先修接口、保持传感信息一致，再做反馈频率对照；加入接触信息属于独立消融。
7. **评价标签的局限。** 当前 `survived` 仅检查 `termination_reason != fell`，left 在倾斜 82.2°主动放弃时仍得到 1，不能视为任务成功。give_up 会执行一帧 stop action 后再终止；forward/turn 在该帧跨过 >90°阈值，最终标签为 fell。后续应分别保存主动停止意图与物理终止事实，并增加完成 horizon 等指标。
8. **不能声称更大预算导致退化，也不能声称达到 SOTA。** 每场景仅一次且模型采样不固定；预算、提示 `calls_left` 和采样同时变化。原生 ONNX 是 bundled pretrained baseline，Astra 与它还存在动作约束和反馈频率差异。

分析产物保存在 `outputs/analysis-20260910-parallel/`：report、return-comparison、reward-decomposition、trajectories、policy-actions（PNG/SVG）、完整分项表、诊断 JSON、20 条回放轨迹和源日志。原始产物保留，后续更正以本记录和 issue 更新为准。

## 2026-09-10：动作执行反馈 v2 修复

实现提交：[e3116d7](https://github.com/guajun/gpt-dog-eval/commit/e3116d7710e262858c65aeba4e73a94c45d1137c)。验证在提交前的同一源代码上完成；测试日志仍记录旧 HEAD 加 dirty 标志，不将其伪装成提交后的执行。

本次改变的是执行接口，不增加接触、足端位置、reward 观测或确定性稳定器。

- 保留 ONNX 的 `policy_obs` 48 维向量逐值不变，包括既有动作历史语义。
- embodiment 在物理步成功后，单独提供 post-controller/post-guardrail 的 `last_applied_action`，reset 为零；这是执行回执，不是新传感器。
- agent 用实际动作进行新 chunk 插值、slew 校验、hold、give_up 和预算耗尽时的 stop action。缺少新字段直接报错，不退回已知滞后的旧字段。
- 提示将旧字段改称 `actor_previous_action`，明确它比当前目标早一控制步；其余任务目标、工具定义、动作范围、预算 60 和最长 15 步保持原设置。slew 浮点容差为 1e-7，忽略 float32 表示噪声。
- 每个运动 chunk 回执保存 requested/executed steps、completed/interrupted、反馈完整性、实际末动作、修改的步/关节及 proposed/applied 值和最大误差；`completed` 仅代表步数完整，不代表目标未经改写。
- 终止 hook 从实际 StepRecord 刷新最终回执与 transcript；请求 15/执行 10，以及异常退出的部分轨迹继续保留。metadata 写入 `action_feedback_version=2`。
- 此次不改变原始 reward、物理终止条件或既有 `survived` 定义；它们的指标改进见后续清单，避免与接口修复混成一个实验变量。

验证：本地和远端各 29 项 pytest 通过，Ruff/mypy 通过。首次真实 MuJoCo 检查运行 3,032 步，保留在 `outputs/action-feedback-v2/`；随后补齐标准 JSON sink 并加入故障注入，完整验证运行 3,042 步，保存在 `outputs/action-feedback-v2-logged/`。

完整验证包含 Zero、原生 ONNX、Fake 各四场景 250 步，所有 return 与历史值逐值一致；非零 ramp/reversal/hold 测试无实质性执行偏差；外层 limiter 单独收紧到 0.025 后，3 个被改写控制步准确进入回执。刻意在第 10 步后注入 simulator fault，最终错误日志仍保留 requested=15、executed=10、interrupted、实际末动作和错误原因。Fake 四场景最后的 15/10 回执也全部通过检查。验证不产生付费推理调用。

完整验证日志位于 `outputs/action-feedback-v2-logged/`，同时有 `verification.json` 汇总及逐步实际动作 JSONL：

| 子目录 | 日志 ID | 实际步数 | 结果 |
|---|---|---:|---|
| zero | `1ace3137` | 1000 | 四场景完成，与历史 return 相同 |
| onnx-native | `4b70bf5e` | 1000 | 四场景完成，与历史 return 相同 |
| fake | `4f2b1110` | 1000 | 四场景完成，与 Zero 相同，全部执行回执完整 |
| scripted | `ec392cf8` | 16 | 无实质性改写，末 chunk 15/10 |
| scripted-strict-guardrail | `bca8fcfa` | 16 | 3 个改写控制步进入回执，hold 保持实际目标 |
| intentional-fault | `686a6f58` | 10 | 预期 error，部分执行与异常原因完整保存；不是意外失败 |

复现命令：

```bash
uv run pytest -q
uv run ruff check src tests scripts/verify_action_feedback.py
uv run mypy
JAX_PLATFORM_NAME=cpu uv run python scripts/verify_action_feedback.py \
  --output outputs/action-feedback-v2-logged
```

修复后尚未追加真实 Astra 评测，因此当前只能确认执行接口修正，不能报告模型成绩提升。

## 后续实验：按顺序追加结果

- [x] 建立无头 CPU 环境、Zero/ONNX baseline 和免费 provider 链路。
- [x] 记录全部真实试跑、超时、调用预算耗尽、主动放弃与摔倒。
- [x] 完成四路 60-call 独立上下文批次及完整 reward/policy review。
- [x] 修复实际 action 反馈，验证原始 PPO 观测与 baseline 不变，并验证异常日志。
- [ ] **A：修复后 Astra 同协议复测。** 同四场景、250 步、预算 60、max chunk 15、xhigh、eval_seed=0、独立上下文；标注 action feedback v2。保留全部尝试，记录 return、执行步数、停止原因、calls/tokens、改写比例；与 v1 比较时承认模型采样差异，不把一次重跑当因果证明。
- [ ] **B：统一补足指标。** 同时报告完成 horizon、物理 fall、主动 give_up/预算耗尽、XY/yaw RMSE 和倾角；沿用原始 return，历史轨迹可离线重算，区分旧版 survived。
- [ ] **C：保持传感信息一致，比较反馈频率。** 预先固定如 1/5/15 步 chunk 条件并允许动态 horizon；预算应足够覆盖各自最坏调用数（1 步方案至少 250 次），否则是在比较调用上限。明确动作限幅和 ONNX 原生/受限版本；记录成本和 wall time。
- [ ] **D：多 seed、多次推理重复。** 预先固定 seed、重复次数和汇总规则，保留每次失败，不只挑最好轨迹；报告完成率、分布/置信区间与完整 trial 清单。
- [ ] **E：可选接触信息消融。** 将 contact/足端位置等增强输入单列命名，与同信息条件区分，不能反推没有 contact 的模型必然失败。
- [ ] **F：可选 chunk 提前中断或混合低层控制。** 单列 policy 名称；使用稳定器后的成绩不能叫纯 GPT 直接运控。

后续每次实验/修复，在同一 issue 追加一条带时间的记录，同时更新本文件：假设与唯一主要变量；代码 commit/dirty 状态及依赖；场景/seed/horizon/预算/工具限制/provider/model；完整 log ID；结果、成本、失败原因和相对前一批的解释；下一步。公开记录不包含 API key、secret.toml 内容、私有 endpoint 或 SSH 配置。原始日志与大体积图表按忽略规则留在产物目录，issue 记录可定位的文件名与汇总。
