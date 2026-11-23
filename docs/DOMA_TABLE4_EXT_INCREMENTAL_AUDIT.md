# 原 DOMA 仓库增量部署：差异核查与待确认方案

审查日期：2026-10-13。
唯一部署目标：`/home/fkk/my_data/disk2/code/DOMA`。
本轮仅进行只读服务器审计和本地文档/清单准备。未上传实现，未启动 GPU 检查或训练，未删除 worktree。

## 1. 实际 Git 差异

原 DOMA 仓库与 `DOMA-table4-ext` 的 HEAD 均为：
`faa096039531fc0875c6a9bf250b9432b4e86a4e`。

核查了分支间 diff、两个工作区的 unstaged/staged diff 和未跟踪文件。
两边没有已提交差异，也没有未提交源码修改。独立 worktree 的忽略项为
`dataset -> /home/fkk/my_data/disk2/code/DOMA/dataset` 软链接。
本地待部署内容从未进入该 worktree，因此不需要把整套代码目录复制回来。

另外逐文件核对两边 SHA256：

| 文件 | 原仓库与 worktree |
|---|---|
| opencood/models/sub_modules/doma_object.py | 相同 |
| opencood/models/sub_modules/doma_config.py | 相同 |
| opencood/loss/doma_object_loss.py | 相同 |
| opencood/tools/train_ddp.py | 相同 |
| opencood/tools/train_utils.py | 相同 |
| opencood/tools/inference_doma_heter_in_order.py | 相同 |
| opencood/tools/doma_tools.py | 相同 |

原仓库中新增配置目录、扩展结果目录和六个 Python/Shell 目标文件尚不存在。
读取原始审计证据：本地 `opencood/logs/temp/ext_implementation/incremental_remote_audit.json`。
清单中的哈希属于本次核查时点；实际部署前需要重新检查，不能覆盖后来出现的文件。

## 2. 消融本身不要求修改运行时模型

现有配置已经能够独立表达全部四组：

| 配置组 | M | G | C | 运行时代码修改需求 |
|---|---|---|---|---|
| G_Only | 0 | 1 | 0 | 无 |
| C_Only | 0 | 0 | 1 | 无 |
| G_C | 0 | 1 | 1 | 无 |
| M_C | 1 | 0 | 1 | 无 |

M 对应 `multi_granularity.enabled`；关闭时保留 Detail 对象分支。
G 对应 `geometry.enabled`；关闭时 refiner 的 32 维几何输入保持为零，网络宽度不变。
C 对应 `consensus.enabled`；关闭时使用现有首成员残差及无效回退规则。
因此没有“某个消融因素无法通过配置实现”的情况，也没有修改 `doma_object.py` 的理由。

## 3. 本地实现包含的运行时扩展：需要明确确认

虽然旧文件均未修改，但本地新增的 `train_table4_ext.py` 确实是训练运行时代码，
不能把整个待部署包称为纯 YAML 与外部调度。
它解决的是完整训练状态恢复，而不是 M/G/C 的表达问题。

历史 `train_ddp.py` 的 model_dir 恢复只读取模型权重和 epoch，不恢复 Adam、scheduler、
随机数流、bestval 状态或 DataLoader generator。直接调用该入口并不能承诺完整 epoch 续训。

推荐在原仓库**仅新增**以下运行时文件，保留现有训练入口不变：

| 新增文件 / 函数 | 最小行为变化 |
|---|---|
| opencood/tools/train_table4_ext.py / main | 从原训练器派生；新增 restore_epoch/save_epoch 调用；每轮日志补充参数组 weight_decay；由总控负责正式推理，取消训练器末尾自动普通 inference；正常结束关闭 writer/DDP |
| opencood/tools/table4_ext_state.py / save_epoch, restore_epoch | 原子写入/恢复模型、Adam、scheduler、可选 AMP、bestval、每 rank RNG/buffers、DataLoader generator、数据重排状态 |
| 同上 / load, atomic_save, rng_state, set_rng, fingerprint | 上述恢复功能的读写、原子保存和一致性校验辅助函数 |

专用训练器相对于原训练器的完整 patch 已存在：
`opencood/logs/temp/ext_implementation/trainer_diff.patch`。
调度器 `table4_ext.py::train_stage` 显式选择该新增入口，原训练器不导入这些新文件。

这会带来一份额外训练入口的维护成本，但不需要第二套 DOMA 代码目录。
如不接受新增训练运行时文件，可另行将调度器改为直接调用原 `train_ddp.py`；
代价是只能安全跳过已完成阶段，未完成阶段需保留现场后从头执行，
不能把模型权重加载伪装成完整 Adam 断点恢复。此替代方案目前尚未改写实施。

按用户第 8 条，本轮停在可审阅方案，等待是否接受上述新增训练入口的确认；
未改动或部署任何原模型/训练/推理文件，也未投放缺少训练依赖的半套启动脚本。

## 4. Full DOMA 与历史实验的保护

不改动任何原模型、loss、训练器、推理器、组装器、正式 YAML 或数据 assignment。
原 Full DOMA 命令继续调用原入口，原 checkpoint 参数名及前向路径保持不变。
新增入口仅供扩展脚本显式调用，不替换模块名，不猴子补丁现有训练器。
部署后应重新比对原 tracked 文件与部署前 commit/diff，确保变化只为新增文件。
这些措施保持源码和协议复现条件；不宣称 CUDA 数值位级确定性已经验证。

不把本地 `.gitignore` 改动同步到服务器。日志目录内的主入口可后续单独 `git add -f`，
无需为新增脚本改动原仓库的任何既有 tracked 文件；本轮不执行 Git add/commit/push。

## 5. 历史配置复用与训练范围

本次重新读取 Row1/Row2/Row3 的 15 份实际 `config.yaml`，与本地历史快照进行语义比较。
四组的 20 份 YAML 继续采用 Row2_DualLayer 的五份真实保存配置作为唯一基准：
只改变实验名称和相应 M/G/C 开关，其他字段逐项保持一致。

| 阶段 | epochs 字段 | 每卡 batch | LR | eval/save |
|---|---|---|---|---|
| Stage1 m1 | 30 | 4 | 0.002828 | 2/2 |
| Stage2 m2 | 25 | 4 | 0.0012 | 1/1 |
| Stage2 m3 | 25 | 8 | 0.0030 | 2/2 |
| Stage2 m4 | 25 | 4 | 0.0017 | 2/2 |

seed=303、Adam、scheduler、loss weights、数据划分、预处理、范围、assignment、
候选与后处理均继承 Row2。Stage2 沿用历史初始化 epoch=1 的语义，实际执行 index 1 至 24。
Row3 的 m2 eval/save=2/2，与 Row2 不同；Row1 m2 和 Full Stage2 的预算也不同。
这些差异保留在审计中，不通过重新训练已完成行来“修正”历史。

**不重新训练 Row1、Row2、Row3、Full 或基础对象分支。** 调度顺序只包含
G_C → M_C → G_Only → C_Only 四个此前未完成的组合。

已有训练权重不能仅因形状可加载就视为可复用：

| 新组合 | 与已有组合比较 | 不直接复用已训练 Stage1 的原因 |
|---|---|---|
| G_Only | Row1(000) / Row3(110) | 前者改变 G，后者改变 M，训练图及参数归属有差异 |
| C_Only | Row1(000) | C 改变 Stage1 共识监督，尚无等价训练轨迹证明 |
| G_C | Full(111) | 关闭 M，Context 模块、有效性掩码和表示均改变 |
| M_C | Row2(100) | C 改变 Stage1 共识监督；Stage2 虽不计算该项，仍依赖不同的 Stage1 anchor |

因此计划训练四个新组合自己的 Stage1 和 m2/m3/m4，属于新实验训练，不是重跑旧配置。
Stage2 使用同组 Stage1 bestval 初始化。成功完成的扩展阶段未来也按配置与产物哈希校验后跳过。
本轮不加载全部历史权重做昂贵验证，也不声称已证明旧 checkpoint 与新组合等价。

## 6. 增量文件清单与部署顺序

待确认的实现包共 28 个文件，保持仓库内原有层级：

1. `opencood/hypes_yaml/opv2v/MoreModality/DOMA/V2_R2_TABLE4_EXT/{G_Only,C_Only,G_C,M_C}/` 下各 5 份 config.yaml，共 20 份。
   相对阶段路径：stage1、stage2/m2、stage2/m3、stage2/m4、final_infer。
2. 同级 `history_snapshot.json`：1 份历史依据。
3. `opencood/logs/temp/run_doma_table4_ext_ablation.sh`：用户入口。
4. `scripts/run_doma_table4_ext_ablation.sh`：后台与锁管理。
5. `opencood/tools/table4_ext.py`：调度、配置核对、状态、组装与汇总。
6. `opencood/tools/check_table4_ext.py`：验证辅助。
7. `opencood/tools/train_table4_ext.py`：新增训练入口，待确认。
8. `opencood/tools/table4_ext_state.py`：新增状态恢复辅助，待确认。
9. `docs/DOMA_TABLE4_EXT.md`：使用说明。

本轮另生成本审计文档和 `docs/DOMA_TABLE4_EXT_INCREMENTAL_MANIFEST.json`，
后者包含上述 28 个文件的准确相对路径、字节数、SHA256 和部署分类。

确认后的部署流程：

1. 再次确认原仓库 HEAD、dirty 状态与所有目标冲突；任何不同内容的已有目标一律停止，不覆盖。
2. 只新增清单中的文件，逐项核对哈希；不复制整个仓库，不同步 .git、dataset 或旧 logs。
3. 原历史实验目录保持只读，扩展实验产物仅写新根目录
   `opencood/logs/DOMA_V2_R2_2GPU_TABLE4_EXT_ABLATION_SEED303/`。
4. 执行语法、配置核对和无 GPU 的 check/status；完整服务器 preflight 留给显式启动前。
5. 核对旧 tracked 文件无变化，再报告增量已部署。本轮不进行此部署步骤。

## 7. 一键入口与后台行为

最终目标路径（本次尚未部署，不是已存在的服务器入口）：
`/home/fkk/my_data/disk2/code/DOMA/opencood/logs/temp/run_doma_table4_ext_ablation.sh`。

```bash
cd /home/fkk/my_data/disk2/code/DOMA
bash opencood/logs/temp/run_doma_table4_ext_ablation.sh check
bash opencood/logs/temp/run_doma_table4_ext_ablation.sh start
bash opencood/logs/temp/run_doma_table4_ext_ablation.sh status
bash opencood/logs/temp/run_doma_table4_ext_ablation.sh resume
```

物理 GPU 固定 0、2，逻辑 GPU 为 0、1，DDP `--nproc_per_node=2`。
保留 GPU 空闲/UUID 校验、tmux、双重启动锁、阶段标记、checkpoint SHA256、
m2→m3→m4→m1 权重组装、CAV1–CAV4 顺序推理及 CAV4 结果汇总。
tmux 支持 SSH 断线继续，不承诺服务器重启自动恢复。
`check` 的完整模式含短 CUDA/DDP/真实 batch 检查；本轮不执行。
本轮只使用 `check --local` 做静态协议审查，不把它称为服务器可运行性通过。

## 8. 独立目录保留结论

当前 `DOMA-table4-ext` 没有独有提交或未提交源码，仅有 dataset 软链接。
但增量文件尚未完成原仓库部署与部署后校验，因此目前不进入删除步骤。
确认增量安全部署后，可再次核查其 HEAD/dirty/ignored 状态，再报告是否可移除。
本轮未删除 worktree、分支或软链接。
