# 81 字基线的 Colab 准备与运行

## R16同分布基线（已准备）

复用第二轮训练核心，另存 [hanzi1140_r16_e10.ipynb](hanzi1140_r16_e10.ipynb)，使用独立的 `hanzi1140_r16_data.zip` 和 `MyDrive/hanzi1140_r16/`。
模型为随机初始化的1,140类ResNet18，sigma=0，中间分辨率16，完整基线10轮；只保留准备、训练、报告三个单元。
可读性以用户人工确认为依据，停止追加盲认与评分。R16数据包已生成（约1.43 GB）。将其上传到Drive的 `MyDrive/hanzi1140_r16/`，在Colab上传新Notebook并顺序运行三个代码单元即可。旧满分基线文件保留。

## 第二轮完整 1,140 字数据（2026-10-29）

全部 443 个来源字组涉及的 1,140 字已生成 656,640 张图，并通过独立像素审计。
新入口为 [`hanzi1140_round2_e10.ipynb`](hanzi1140_round2_e10.ipynb)：沿用首轮 ResNet18，完整基线训练 10 个 epoch。
这里将用户“训练10层”按训练 10 轮理解，网络层数继续沿用首轮。

1. 在 Drive 的“我的云端硬盘”新建 `hanzi1140_round2`，上传原数据包 `colab/dist/hanzi1140_round2_data.zip`。
2. 把新 Notebook 上传到 Colab、保存 Drive 副本，选择 Python 3 与 GPU。
3. 顺序运行六个代码单元。第④格做独立小集 1 轮短跑；第⑤格默认完成完整基线 10 轮。
4. 断线后重跑准备格和同一训练格，从最近完整 epoch 恢复。默认一次跑完；设 `EPOCHS_THIS_SESSION=2` 可分段运行，总预算仍为 10。

源码已内嵌，沿用首轮交叉熵、AdamW、batch64、lr=0.001、weight_decay=0.0001、seed42 和 AMP。
Drive 输出为 `hanzi1140_round2/runs/baseline1140_e10_s42_v1/`，包括 best/last.pt、配置与每轮日志、曲线、两个开发集的混淆矩阵及预测。
第⑥格在完整 10 轮后自动评估 val_select 与 val_graph，test 保留。
本机字组浏览入口为 `training/data/round2-1140-v1/preview/index.html`；详细协议与验收见
[第二轮记录](experiments/round2_1140_v1/README.md)。2026-10-29 用户回传已完成 10 轮训练，两个开发集各 109,440 张均为 100% 准确率；
详见[回传结果](experiments/round2_1140_v1/reported_results_2026-10-29.json)。尚未独立读取 Drive 权重或完整 history，test 未评价。

## 退化标定第一阶段（2026-10-20）

当前新阶段入口为 [`hanzi81_degradation_phase1.ipynb`](hanzi81_degradation_phase1.ipynb)。
上传该 Notebook 到 Colab，顺序运行两个代码单元：准备固定标定集 → 对原第 38 轮 `best.pt` 做 25 档推理并保存报告。
源代码已嵌入，沿用原 Drive 的数据包与模型，不必上传新 ZIP 或运行旧 Notebook。
标定集为 81 字 × 16 个原 `val_graph` 清晰位移 × 25 档 = 32,400 张；本地数据及 54 对预览已生成。
2026-10-27 已在本机 CPU 完成原模型及 PaddleOCR 的 25 档扫描，指标与逐图预测已归档；该 Notebook 保留为 Colab 复现入口。
完整协议、结果状态和本地验证见[第一阶段记录](experiments/degradation81_phase1_v1/README.md)。

## 已完成基线后的会话恢复版（2026-10-13）

已有 `baseline81_s42_v1` 和学习曲线准备文件时，上传
[`hanzi81_baseline_new.ipynb`](hanzi81_baseline_new.ipynb)。它根据用户从 Colab 导出的运行状态整理，
包含三个可直接使用的代码单元；源码已嵌入，不需要另外上传 `session_cells`。

1. **初始化与恢复**：挂载 Drive，复用或恢复固定包，校验文件，从已冻结实验目录恢复学习曲线脚本，并检查原 GPU 环境。不会自动训练。
2. **分段续训**：默认 `RUN_TRAINING=False`；开启后一次选择一个档位，默认追加 3 块。每档总预算仍为 10 块 / 2430 批次，原脚本负责严格续训。
3. **检查与汇总**：可在新 CPU 会话独立运行，从 Drive 读取日志、指标和计时报告。部分进度不算最终结果，四档完整后才生成固定预算比较图。新报告保存到 `analyses/baseline81_s42_v1/session_reports/`。

第 6 项已接入第③格：复用原三轮计时报告，校验后计算吞吐、阶段耗时占比、显存容量占比、检查点空间及预算估算。
成功时保存 `efficiency_resources.json` 和 `efficiency_resources.png`，状态为 `complete_saved_measurement_analysis`；
报告缺失或不一致时仍为未完成。仅读取及分析已有测量，无需 GPU，不重跑短跑/拟合/基线，不自动执行模型评估。
完整原始代码和运行输出保存在
[`hanzi81_baseline_new.before_session_refactor.ipynb`](hanzi81_baseline_new.before_session_refactor.ipynb)，供追溯，不应全部执行。
续训前选择与原实验匹配的 GPU；仅读报告可直接运行第③格，无需准备本地训练数据。

维护入口为 `session_cells/*.py` 和 `make_session_notebook.py`；离线验证命令是
`python -m unittest colab.test_efficiency colab.test_session_notebook -v`。此生成器不重建原始数据 ZIP。
以下章节仍是首次建立基线的旧 Notebook 操作说明。

2026-10-12：用户已选择 Colab，已有 Pro／算力额度，并确认先训练 81 字基线。
完整 1,140 字候选池继续保留。本轮先建立可恢复的普通交叉熵基线，不实施针对性损失和四组正式对照。

## 先上传两个文件

1. 打开 [Google Drive](https://drive.google.com/)，在“我的云端硬盘”建立 `hanzi81` 文件夹。
2. 把本地 `colab/dist/hanzi81_colab_bundle.zip` 上传到该文件夹；保持这个文件名。
3. 打开 [Google Colab](https://colab.research.google.com/)，选择“文件 → 上传笔记本”，上传 `colab/hanzi81_baseline.ipynb`。
4. 选择“在云端硬盘中保存副本”，命名为 `hanzi81_baseline`。Notebook 可保存在 `hanzi81`，其位置不影响代码里的数据路径。
5. 菜单“运行时／代码执行程序 → 更改运行时类型”，选 Python 3 和 GPU。先选界面中可用的 T4；无需预先选择高 RAM 或最昂贵的 GPU。
6. 逐格运行 Notebook。弹出 Drive 授权时，由你登录并完成授权。不要一开始就改参数或跳过检查。

本地上传包只包含本研究的代码、81 字表、已固定字体和许可、四份数据及说明；不包含其他课程文件、Git 历史、本地环境或账号凭据。
无需 GitHub 仓库、SSH 密钥或访问令牌。

## Notebook 运行顺序

| 步骤 | 你应看到什么 | 何时进入下一步 |
|---|---|---|
| 1 GPU 检查 | GPU 型号、显存、PyTorch 版本和 GPU 前向输出 | `torch.cuda.is_available()` 为 True |
| 2 Drive 挂载 | 上传包路径与结果目录 | 没有授权或文件不存在错误 |
| 3 复制与校验 | 四集合数量与 46,656 个唯一像素哈希 | 全部文件 SHA-256 校验通过 |
| 4 训练样例 | 白底黑字的 64×64 图 | 没有空图、缺字或明显裁剪 |
| 5 一轮短跑 | 一个 epoch 和 Drive 中的 best/last.pt | 训练、验证、保存都成功 |
| 6 小集拟合 | 每类 2 图、共 162 图的 20 epoch 曲线 | loss 下降，训练子集准确率趋向拟合；否则排查 |
| 7 完整基线 | 把 START_BASELINE 改为 True，先完成 2/40 epoch | 检查耗时、loss 和算力消耗后继续 |
| 7 继续训练 | 把 EPOCHS_THIS_SESSION 改为 None，重复运行同一格 | 从第 3 轮继续，计划仍是 40 epoch |
| 8 开发报告 | 曲线、val_select 分项、误认字对 | 用于开发分析；不作为独立测试结果 |

首轮反馈信息：GPU 型号、每轮含保存的耗时、train/val loss 与 accuracy、Drive 中是否存在 `last.pt`。
40 epoch 是初始开发预算，不能仅凭达到轮数就断言收敛。

## 数据与模型

四个集合沿用项目设计书的共享条件表规则；先对所有字符去重，再分配条件，不按图片随机打散样张。

| 集合 | 每字 | 81 字总量 | 用途 |
|---|---:|---:|---|
| train | 192 | 15,552 | 梯度更新 |
| val_graph | 96 | 7,776 | 基线确定后固定混淆关系；当前 Notebook 不自动使用 |
| val_select | 96 | 7,776 | 基线检查点与开发参数选择 |
| test | 192 | 15,552 | 协议冻结后的最终评价；当前 Notebook 不自动使用 |

固定思源黑体 SC Regular、64×64 灰度、四倍渲染、全字表共同基线与字号、±3 px/0.25 px 网格、sigma=0/0.5/1.0。
当前使用 `baseline81-v2`：每个集合／sigma 的横纵偏移均值绝对值不超过 0.25 px，范围两侧均覆盖至至少 ±2.5 px。条件分配只按几何量确定，不使用模型或测试成绩。
归一化为 `(x / 255 - 0.5) / 0.5`，无额外在线增强。生成器发现同标签像素重复时，会为全字表替换该条件；跨标签重复则停止并报错。
标签与原始 `pilot/characters.json` 保持一致，类别 ID 为 0～80。

ResNet18 使用灰度 3×3、stride=1 首层并移除首层最大池化；随机初始化。AdamW、交叉熵、batch=64、lr=0.001、weight_decay=0.0001，余弦降至初始学习率的 1/100。
CUDA 训练使用混合精度，验证用 fp32。数据种子为 20261012，首轮训练种子为 42。

同一字体的亚像素变体高度相关。本轮只能评价固定生成机制下未训练过的条件组合，不能当成真实拍照 OCR 或跨字体泛化。
81 字都在预设字对端点集合中，没有独立的“普通字”对照；不声称已验证对普通字无损。

## 保存和恢复

运行时工作目录是 `/content/hanzi81-<包哈希前12位>`，随 Colab 虚拟机回收而消失。
持久结果是 `MyDrive/hanzi81/runs/<run名称>/`：

- `config.json`：模型、超参数、代码／数据哈希、实际 PyTorch/CUDA/设备信息。
- `history.json`：每个完整 epoch 的训练／验证指标与计算耗时。
- `best.pt`：按 val_select macro accuracy 选择；并列选 CE 低者，再并列保留早期轮次。
- `last.pt`：模型、AdamW 状态、scheduler、AMP scaler、随机状态和训练历史。
- `data_manifest.json`、`labels.json`：对应数据配置、字体与标签证据。
- `source_bundle_sha256.txt`、`environment-pip-freeze.txt`：源包标识和环境记录；固定源 ZIP 另保存在 `hanzi81/packages`。
- `evaluation_val_select/`：指标、81×81 混淆矩阵、54 个预设字对误认统计及逐图预测 CSV。

每轮先写临时文件，再替换最终文件。只从最近完整 epoch 恢复，不从半个 batch 恢复；写 Drive 时遇到断网仍可能失败，必须看见“Checkpoint saved”并确认文件存在。
准备至少约 1 GB 可用 Drive 空间，覆盖上传包、源包副本和三个 run 的检查点；实际用量以 Drive 为准。

断线后：重新连接 GPU → 重新挂载 Drive → 重跑复制／校验和辅助函数格 → 同一 `BASELINE_NAME` 运行训练格。
脚本自动加 `--resume`，并拒绝在已有 run 中从头覆盖。不要同时在多个运行时写同一 run。
恢复会严格核对数据、代码、训练预算、GPU 型号与主要软件版本；变化时恢复原配置，或者另建 run 并重新训练。跨设备、跨版本不保证逐位复现。

## 常见问题

- **没有 GPU**：更改运行时类型并重连；仅在菜单选 GPU 并不证明模型已使用 GPU，第 1 格会实际检查。
- **CUDA out of memory**：重启运行时释放显存，将辅助函数里的 batch size 从 64 改为 32，并给 smoke/overfit/baseline 都使用新 run 名；记录新配置。不要中途改旧 run。
- **Drive 找不到 ZIP**：路径必须是“我的云端硬盘/hanzi81/hanzi81_colab_bundle.zip”，等待上传完成。
- **数据／文件哈希不匹配**：重新上传原始包；不要绕过校验后续训。
- **恢复时报配置改变**：先检查 GPU、包版本、batch、epochs、源码和源 ZIP 是否一致。改变训练配置应新建 run。
- **训练 loss 不降**：先查看第 6 步小集拟合与标签；保留日志，不直接扩大字表或引入针对性方法。
- **接近满分**：如实记录，说明当前开发条件可能过于容易；不故意削弱基线制造混淆。
- **停止使用 GPU**：确认 Drive 文件保存后，选择“断开连接并删除运行时”；关闭网页不等于主动终止运行时。

Colab GPU 类型、可用时长和限制随资源与额度变化，Pro 也不保证固定资源。请以账户当前显示和 [Google 官方 FAQ](https://research.google.com/colaboratory/faq.html) 为准。
官方也建议减少 Drive 读写次数，将归档文件复制到运行时本地再使用。本项目采用这一方式。

## 后续命令与复现

基线充分训练并确定后，才单独进行 val_graph 分析：

```bash
python -m training.train evaluate --data training/data/baseline81-v2 --run-dir /path/to/run --split val_graph
```

最终测试须先冻结实验协议、预算和模型选择，再显式执行：

```bash
python -m training.train evaluate --data training/data/baseline81-v2 --run-dir /path/to/run --split test --final-test
```

不要根据 test 结果继续选参数或挑字。当前没有自动实现针对性损失、3 种子主对照或最终测试，这些在基线开发后另行开展。

数据已打包，通常无需重新生成。若复现生成器，在独立 Python 环境安装 `training/requirements-colab.txt` 和 NumPy，保留版本／FreeType 记录，并运行：

```bash
python -m training.prepare_data --output training/data/baseline81-v2
python -m unittest training.test_pipeline -v
python colab/make_notebook.py
python -m colab.build_bundle
```

生成器拒绝混用不同环境或代码的旧输出，改变渲染环境时要用新输出目录，再重新审核图像、哈希和划分。
训练保存方式参考 [PyTorch 官方检查点说明](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html)。

扫描 PDF 场景下的 PaddleOCR、人工字框复核与 Edge 对照实验，见 [第一阶段方案和操作步骤](experiments/ocr_edge_pilot_v1/README.md)。
