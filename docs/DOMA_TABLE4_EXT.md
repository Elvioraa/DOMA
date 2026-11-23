# DOMA V2 R2 扩展消融：本地交付与历史审计

准备与部署日期：2026-10-13。用户已批准 28 文件原仓库增量部署，新增训练恢复入口包含在批准范围内。
唯一部署目标 `/home/fkk/my_data/disk2/code/DOMA`；不维护第二套代码目录。
已完成部署与 CPU 验证；完整 check 因 GPU2 被其他任务占用而未通过，禁止将其描述为可直接开训。
本轮未执行 start/resume，未运行四组正式实验。最新服务器验收结果见文末。
最终 GPU 指定为物理 **0、2**（用户更正了附件中的 0、1）。

## 文件与使用

主入口：`opencood/logs/temp/run_doma_table4_ext_ablation.sh`。
该入口委托版本控制内的 `scripts/run_doma_table4_ext_ablation.sh`。
完整配置目录：`opencood/hypes_yaml/opv2v/MoreModality/DOMA/V2_R2_TABLE4_EXT/`。
每组包含 `stage1/config.yaml`、`stage2/m2/config.yaml`、
`stage2/m3/config.yaml`、`stage2/m4/config.yaml`、`final_infer/config.yaml`。

| 顺序 / 实验 | M | G | C | Stage1 | Stage2 m2/m3/m4 |
|---|---|---|---|---|---|
| G_C | 0 | 1 | 1 | 重训 | 各自重训 |
| M_C | 1 | 0 | 1 | 重训 | 各自重训 |
| G_Only | 0 | 1 | 0 | 重训 | 各自重训 |
| C_Only | 0 | 0 | 1 | 重训 | 各自重训 |

没有证明旧阶段与新配置在计算图、监督项、参数归属及训练轨迹上等价，故不复用旧训练结果。
各 Stage2 仅初始化自同组 Stage1 的唯一 bestval，彼此独立。
沿用历史入口语义：Stage2 的初始化权重命名为 `net_epoch1.pth`，
训练循环从 epoch index 1 开始，至 24（配置 `epoches: 25`）。
这意味着 Stage2 实际执行 24 个 epoch，与历史脚本保持一致；不是从 0 开始的 25 轮。
初始化用普通副本，不使用可能被 checkpoint 写入操作穿透的软链接。

28 个新增文件已同步到目标服务器原仓库，使用以下命令；本次未执行 start/resume。
部署目标为原 DOMA 仓库，直接使用其既有数据集和 heter_modality_assign；不再部署到独立 worktree。

```bash
cd /home/fkk/my_data/disk2/code/DOMA
bash opencood/logs/temp/run_doma_table4_ext_ablation.sh check
bash opencood/logs/temp/run_doma_table4_ext_ablation.sh start
bash opencood/logs/temp/run_doma_table4_ext_ablation.sh status
bash opencood/logs/temp/run_doma_table4_ext_ablation.sh resume
```

脚本默认使用历史解释器 `/home/fkk/my_data/disk1/miniconda3/envs/DOMA/bin/python`。
如需另一个安装位置，可以设置 `DOMA_PYTHON`，但服务器检查要求环境名为 DOMA。
`start` 和 `resume` 均先检查，再创建 tmux 后台会话。支持 SSH 断线和关闭终端；
服务器重启后不会自动恢复，需要重新执行 `resume`。

Windows 本地检查（不检查服务器 CUDA/数据）：

```powershell
python -B -m opencood.tools.table4_ext check --local
python -B -m opencood.tools.table4_ext status
python -B -m opencood.tools.check_table4_ext
```

## 固定训练协议

采用服务器实际保存的 **Row2_DualLayer** 五份配置，而不是源码默认 YAML。
`history_snapshot.json` 保存所有 Row1/Row2/Row3 和 Full 的源配置正文，以及推理日志的 AP/数据路径摘录。
它不是完整日志备份，也不包含旧 checkpoint。生成后的配置只允许改变名称和 M/G/C 开关；
每次 check 都与快照逐字段对照，防止训练协议漂移。

| 阶段 | 配置 epochs | 每卡 batch / 全局 batch | LR | eval/save 频率 | MultiStep milestones |
|---|---|---|---|---|---|
| m1 | 30 | 4 / 8 | 0.002828 | 2 / 2 | 15,25 |
| m2 | 25 | 4 / 8 | 0.0012 | 1 / 1 | 10,25 |
| m3 | 25 | 8 / 16 | 0.0030 | 2 / 2 | 10,25 |
| m4 | 25 | 4 / 8 | 0.0017 | 2 / 2 | 10,25 |

共同设置：seed=303，Adam，eps=1e-10，Base weight_decay=1e-4，DOMA weight_decay=0，
MultiStep gamma=0.1，无额外 warm-up。对象损失权重 0.2，个体/共识项权重各 1，quality 权重 0。
保留原始 OPV2V-H 路径、assignment、各模态预处理、训练/验证划分、
通信范围、候选生成、后处理、max_train_proposals=128、max_infer_proposals=64。
推理显式固定历史入口的默认范围 `204.8,102.4`，并按 `[1,2,3,4]` 顺序运行，
仅取第 4 条 AP 作为 CAV4，使用原有评价器。不是从进度条或终端截图猜测 AP。

选择 Row2 是为了保留一份实际完成的单一配置基准，避免混合历史不同协议。
所有新组使用同一套阶段协议，不根据 AP 改学习率、epochs 或 checkpoint 策略。
checkpoint 选择保持唯一 validation-loss bestval，不选择最高测试 AP。

## 历史差异与数值来源

| 历史项目 | 已核实 AP30 / AP50 / AP70 | 协议 / 解释 |
|---|---|---|
| Row1_Base | 0.9041 / 0.8946 / 0.8276 | 对象分支仍启用；不是纯 HEAL。m2 为 30 epochs |
| Row2_DualLayer | 0.8977 / 0.8871 / 0.8154 | M-only；Stage2 配置 25 epochs，m2 eval/save=1 |
| Row3_Geometry | 0.8944 / 0.8820 / 0.8044 | M+G；Stage2 配置 25 epochs，m2 eval/save=2 |
| Full 的 final_infer | 0.9071 / 0.8942 / 0.8275 | 不等于用户提供的正式 Full 数值 |
| Full 的 `m2lr0p0012_m3lr0p0030_m4lr0p0017--final-result` | 0.9128 / 0.9018 / 0.8332 | 用户提供的 Full 数值实际来源 |
| 用户提供的 HEAL | 未核实 / 0.8922 / 0.8103 | 本次未找到并审计该来源；不纳入已核实结果表 |

Row1/2/3 数据来自各自 `final_infer/infer.log` 的四条顺序 AP。
Full Stage2 保存配置为 30 epochs，且 m2/m3/m4 均 eval/save=1。
Full 目录同时存在 LR 组合和早期 epoch 搜索结果；最终 checkpoint 的确切选择链与
最终权重来源尚未做张量/哈希回溯，不能声称与新实验是严格控制变量比较。
旧目录的训练完成文字和 bestval 文件名已读到，但旧权重内容没有完整加载验证。
汇总因此将历史行标记为 `HISTORICAL_LOG_ONLY_PROTOCOL_DIFF`。
与 Full 的差值可以描述结果，不能直接解释成某个机制的净因果贡献。

## 模型语义及代码隔离

没有修改正式模型、损失、merge、推理入口或正式 YAML。
本地基线和已审计服务器 HEAD 均为 `faa096039531fc0875c6a9bf250b9432b4e86a4e`。

- M=0：将 multi_granularity 设置为严格的 `{enabled: false}`，保留 Detail 对象编码及 refiner。
  有效性掩码由 Detail ∧ Context 变为仅 Detail，与现有代码一致。
- M=1：现有 Detail + 可学习缩放的 Context 投影残差融合，不新增研究模块。
- G=0：不实例化 geometry encoder，向 refiner 输入固定 32 维零向量；
  refiner 输入仍为 128+32=160，保留其规模、键名和初始化机制。
- C=0：使用第一个成员（ordered inference 的 ego/m1）的残差；该成员无效则保留原框，
  不选择其他成员或引入新加权规则。Stage1 共识损失关闭，个体损失保留。
- C=1：有效成员的均值共识，无质量权重。Stage2 仍只计算个体适配损失。

新增独立 `train_table4_ext.py` 来源于该 HEAD 的 `train_ddp.py` 副本。
相对原实现仅增加 epoch 恢复状态、每组 LR/weight_decay 日志、显式关闭 writer/DDP，
并移除训练结束后的自动普通 inference（由总控统一运行正式 DOMA 推理）。
Full DOMA 的现有入口和 checkpoint 参数命名保持不变。
`.gitignore` 仅新增主入口文件的白名单，实验产物仍被忽略。

## 失败处理、恢复与可追溯性

输出根目录固定为 `opencood/logs/DOMA_V2_R2_2GPU_TABLE4_EXT_ABLATION_SEED303/`。
每组的 `run_manifest.json` 记录 PENDING/RUNNING/COMPLETE/FAILED、实际开关、
Git HEAD、源文件及配置 SHA256、各次命令/PID/退出码/日志、bestval epoch/loss、checkpoint 哈希。
总目录 `execution_manifest.json` 还记录 Python 路径、Git dirty 状态及数据路径/assignment 指纹。
数据指纹是路径、场景目录和 assignment 哈希，不是假称所有传感器文件的内容哈希。

两个独立文件锁保护启动和 worker；tmux 会话检查辅助防止重复启动。
阶段仅在退出码为 0、预算完整、唯一 bestval 及产物校验通过后完成。
组装顺序固定 m2→m3→m4→m1，复用现有 DOMA 所有权检查，并对最终完整模型 strict load。
失败立即停止后续依赖阶段；各次日志不会被新尝试覆盖。
已有 COMPLETE 阶段只有配置和产物哈希仍匹配才能跳过。

恢复状态 `resume_state.pth` 原子保存：模型、Adam、scheduler、AMP scaler（若启用）、
bestval 权重/指标、每 rank 的 Python/NumPy/Torch/CUDA RNG、模型 buffers、
DataLoader generator、数据重排前 RNG 和下个 epoch。
恢复重建相同数据重排，并重新执行未完成的 epoch；不是 batch 级续训。
若首个 epoch 尚未形成完整状态，则把已有普通 checkpoint 移入该阶段的 `interrupted_*`，
保留日志后重新初始化该阶段，不把只有模型权重的恢复假称为完整 Adam 续训。
恢复时新的未提交 bestval 移入 `interrupted_bestval`，并恢复已提交 bestval。
CPU 一步续训等价测试不能证明 GPU 位级确定性。

每组完成或失败都会刷新 `ablation_results.csv`、`ablation_results.md` 和 `execution_report.md`。
成功的新实验 AP 必须从已保存日志解析；失败/未运行项保持空值，不能沿用旧结果。

## 检查边界

本地验证覆盖：Python/Shell 语法、20 份配置和历史字段一致性、M/G/C 对象分支
forward/loss/backward、Stage1/Stage2 DOMA 参数冻结、对象参数严格组装、
Adam/scheduler/随机状态恢复、AP 拒绝异常输入、控制器模拟成功/失败/跳过/损坏产物、
本地 check/status。验证日志位于本地 `opencood/logs/temp/ext_implementation/`。

本次上述本地检查均 PASS。完整流程模拟验证了四组共 24 个训练/组装/推理步骤、
全部完成后的零重复执行，以及首阶段失败后停止后续步骤。
模拟发现并修复了 Windows CRLF 源配置复制成 LF 导致恢复哈希不一致的问题：
正式配置现在按字节原子复制，完成后复核原始文件 SHA256。
模拟使用临时目录和替代训练进程，不代表执行过 GPU 实验。

改动清单：新增 20 份 YAML、1 份历史快照、4 个独立 Python 文件、
2 个 Shell 入口和本说明文档；仅修改 `.gitignore` 以跟踪主入口。
`git diff --check` 通过。独立训练副本的审阅 diff 保存在
`opencood/logs/temp/ext_implementation/trainer_diff.patch`。
未提交 Git commit 或推送；本轮已按用户批准完成原仓库增量部署。

服务器 `check` 将验证真实依赖/模型/数据构建、100 GiB 磁盘门槛、GPU 空闲状态、
CUDA UUID 物理映射、两 rank NCCL 与 DDP、每组每个训练阶段的一个真实 batch 前后向。
这些短检查不更新优化器、不写正式 checkpoint。任何失败都阻止 start。
真实 DOMA 数据批次和 CUDA/NCCL 恢复本次未运行，不能宣称已验证。
服务器两 rank Gloo 恢复、tmux 断线存活探针及 flock 已验证，具体边界见文末。

范围调整前只读发现：物理 0/2 均为 V100 32GB，当时 GPU2 被 PID 7959 占用约 12 GiB；
这是检查时的快照，不是当前空闲保证。启动时会重新检查，绝不杀其他进程。
范围调整前已创建服务器 `/home/fkk/my_data/disk2/code/DOMA-table4-ext` worktree，
分支 `codex/table4-ext-ablation`，基于原 HEAD，并添加了指向原仓库 dataset 的链接。
独立 worktree 未部署本次实现，也未删除；本次实现仅新增至原仓库。
未启动任何正式训练。正式服务器已有源码及旧实验结果未改动。

## 2026-10-13 服务器部署验收

部署基线 HEAD：`faa096039531fc0875c6a9bf250b9432b4e86a4e`。
原仓库新增 28 个批准文件，不同步本地 `.gitignore`，不覆盖任何原 tracked 文件。
首次部署 28/28 SHA256 全部匹配批准清单。随后仅给 20 份新 YAML 添加历史协议差异注释，
并更新本说明；最终同步状态及 SHA256 以增量部署清单为准。
四个新增 Python 文件及两个 Shell 文件与批准版本保持逐字节一致。
清单分别保留首次批准哈希与最终发布哈希。

验收证据保存在 `/tmp/doma-table4-acceptance-hnpgw411/`。
本地归档目标为 `opencood/logs/temp/ext_implementation/server_acceptance/`。
临时脚本、微型测试 checkpoint 和日志不属于正式实验产物，也不写入旧实验目录。

| 检查 | 本次结果 |
|---|---|
| 28 文件完整性 / 原源码保护 | 部署核验通过；420 个已有 tracked 文件哈希保持不变 |
| 原 Table4 / Full 产物保护 | 10520 个既有文件 size/mtime 纳秒级元数据与部署前基线一致；未向这些目录写入 |
| Shell / Python 静态检查 | 通过 |
| 20 配置 / Row2 实际协议 | 通过；只改变 M/G/C、名称及非语义注释 |
| 服务器真实依赖、模型及训练/验证数据集构建 | 通过；未使用依赖替代模块 |
| 四组对象分支前后向、参数冻结、严格对象权重组装 | 通过 |
| 两 rank CPU/Gloo 原入口对比新增入口 | 通过；Stage1 与 Stage2，DataLoader workers=0 和 2 |
| 模拟中断与完整恢复 | 通过；逐步参数、Adam、scheduler、bestval、采样序列与连续运行一致 |
| 控制器成功/失败/跳过/损坏 checkpoint | 通过模拟验证 |
| 独立 tmux 会话在 SSH 退出后完成 | 通过，已读取 DETACHED_SESSION_COMPLETED 标记 |
| Linux 文件锁重复执行拒绝 / 释放后恢复 | 通过 |
| status | 退出码 0；四组 PENDING，worker lock 为 False |
| 主入口完整 check | 退出码 1；CPU 检查通过后正确阻止占用 GPU2 的启动 |
| CUDA/NCCL / 真实 DOMA batch / GPU 恢复 | 未执行，GPU2 被 PID 7959 占用约 12 GiB |

训练兼容性测试说明：临时测试装置直接执行原 `train_ddp.py` 与新增入口的未改写函数体，
以小型模型、随机数据集和日志装置替代重型模型/数据依赖。
Adam、MultiStepLR、真实 DDP/Gloo、DistributedSampler、全局 validation reduction、
bestval 选择、checkpoint 读写及恢复辅助使用实际实现。
两个阶段、两个 rank、两种 worker 数下，未恢复及恢复后的逐步状态均按零容差比较通过。
服务器旧版 Gloo 不支持 bool 广播，测试装置仅将该通信载荷转为 int32 再转回 bool；
没有修改任何原/新训练器，也没有改变部署端 NCCL 路径。
此测试验证训练控制逻辑和恢复状态，不等于已证明完整 DOMA CUDA 参数更新位级等价。

恢复记录中的 `next_epoch` 与配置 seed、world_size 重建 DistributedSampler.set_epoch，
DataLoader generator 与每 rank RNG 保留各 worker 随机流；`cumulative_steps` 记录已提交更新步数。
模拟中断发生在下一 epoch 的部分更新后，恢复会丢弃未提交 epoch 的进度并重放该 epoch；
已提交 epoch 不会重跑。必须区分这种 epoch 边界恢复与 batch 级精确续跑。
数据集本身及其目录顺序需保持稳定；本次真实 OPV2V-H 数据的中断恢复尚未实测。

本协议没有传入 `--half`，因此 AMP 关闭，恢复状态中的 scaler 为 None。
代码支持启用 AMP 时保存/恢复 GradScaler，但实际启用 AMP 的 CUDA 恢复尚未验证。
CUDA RNG 分支与物理设备 UUID 映射探针同样等待 GPU2 空闲，不能用 CPU 测试替代这些结论。

后续由用户审核后执行一次 start。start 会重新执行完整 check；GPU 被占用时仍会退出，
不会抢占、终止其他任务或自动调整超参数。服务器重启不自动启动实验，须手动 resume。

环境只读检查：PyTorch 2.0.1、CUDA build 11.8、NCCL 支持可用；这不等于 CUDA/NCCL 运行时验证通过。
最终源码复核：420 个 tracked 文件 SHA256 未变，tracked Git diff 为空，独立 worktree HEAD 相同且 status 为空。
SSH 曾间歇超时，之后成功完成上述核验；未修改 SSH 配置。
