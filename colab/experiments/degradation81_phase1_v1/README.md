# 81 字退化标定 · 第一阶段

日期：2026-10-20（Asia/Shanghai）。

**2026-10-27 更新：原第 38 轮模型已在本机 CPU 完成 25 档真实退化扫描，清晰对照 1,296/1,296 正确。**
用户提供的本地权重与归档 SHA-256 一致；每档 1,296 张、共 32,400 次预测，25 个混淆矩阵计数均已核对。
完整结果见 [`local_cpu_20261027/report.html`](local_cpu_20261027/report.html)、
[`summary.json`](local_cpu_20261027/summary.json) 和 [`summary.csv`](local_cpu_20261027/summary.csv)。
推理以 fp32 在 Python 3.13.15、PyTorch 2.14.1+cpu 完成，环境不同于原 Colab T4；不声称跨环境逐位复现。
绘图依赖缺失导致首次报告生成失败，随后安装 Matplotlib，并从已保存预测恢复报告；未重跑模型。

**2026-10-27 PaddleOCR 对照：**使用本机 `PP-OCRv5_mobile_rec` 在相同 25 档上完成 32,400 次识别。
对照逐条核对了原模型输出的 grid、source、条件和标签，每个字在每档均为 16 张。
按识别原始字符串与真实单字完全一致计分，未限制预测属于 81 字表；空输出和多字输出均计错。
清晰对照为 1,277/1,296（98.53%），sigma=2.5、R=16 为 1,187/1,296（91.59%）。
结果见 [`summary.csv`](paddleocr_PP-OCRv5_mobile_rec_cpu_20261027/summary.csv)、
[`predictions.csv`](paddleocr_PP-OCRv5_mobile_rec_cpu_20261027/predictions.csv) 和
[`summary.json`](paddleocr_PP-OCRv5_mobile_rec_cpu_20261027/summary.json)；字对统计见
[`pairs.csv`](paddleocr_PP-OCRv5_mobile_rec_cpu_20261027/pairs.csv)。
通用预训练 PaddleOCR 与本项目的 81 类专用模型训练数据、输出空间和默认预处理不同；此结果不证明差异由某个单独因素造成。

2026-10-20 时本地标定数据、54 对预览和 Colab 推理入口已完成，但原模型的真实退化扫描尚未执行。
当时用户确认 `best.pt` 仅在原 Google Drive 路径；桌面控制接口因 WSL 工作目录兼容问题无法启动。
不将本地测试模型的推理结果记作研究结果，不将人工查看预览记作盲认可读性测量。

## 本阶段要回答什么

原 81 字基线在现有开发条件下满分，缺少支持针对性训练的错误空间。
本阶段固定原模型，测量高斯模糊与低分辨率对开发集准确率、重点字对误认的影响，筛选下一步人工盲认的候选条件。
旧模型在新分布上掉分不能证明同分布训练后仍存在研究空间；本阶段不重训、不确定正式训练难度、不评价原 test。

## 固定协议

| 项目 | 实施值 |
|---|---|
| 来源 | 已归档 `baseline81-v2` 的 `val_graph`，只选 sigma=0 的清晰图 |
| 字表及关系 | 原编号 0～80 的 81 字、原 54 对；无字形重渲染或标签修改 |
| 位移条件 | 所有类别共用 16 个原开发条件，包含居中条件；seed=20261020，仅按几何位置平衡选择 |
| 高斯 sigma | 0、1、1.5、2、2.5，单位为 64×64 输入的像素 |
| 中间分辨率 R | 64、48、32、24、16；完整 5×5 组合 |
| 退化顺序 | 原清晰图 → 白边16 px高斯模糊并裁回 → Lanczos缩至R → bicubic恢复64；R=64跳过缩放 |
| 输入 | 64×64、uint8灰度；模型归一化为 uint8/127.5−1 |
| 数量 | 每档 1,296 张，每字每档 16 张，总计 32,400 张 |
| 原图组标识 | 原 `val_graph` 的 `source_index`；同一来源的 25 个派生版本必须归为一组 |
| 模型 | 原第38轮 `best.pt`，校验已归档指纹；原 ResNet18 构造与权重加载方式 |
| 推理 | eval + inference_mode，fp32，无 AMP、无梯度更新，记录实际环境 |

这一标定集全部属于开发数据。16 个位移不是16个独立字体或真实书写者；不能把32,400张当作独立真实场景样本。
若后续使用这些图进行训练，必须重新设计互斥来源组的划分，不能把当前标定图同时用于训练与最终评估。

## 已实际完成的本地结果

- 32,400 张图均为有效灰度输入，像素哈希全部唯一。
- 空图、跨标签相同像素、同标签重复像素均为 0。
- sigma=0、R=64 档与选定的原清晰图逐像素一致。
- 最外圈最大暗度为 6/255；暗度达到16的边界图片数为0。
- 位移均值：dx=−0.03125 px、dy=−0.078125 px；范围分别为 [−3,2.75]、[−3,3]。
- 已生成全部54对的5×5矩阵预览，并检查简单字与复杂构件的实际图片。
- 预览带有标签，仅用于检查生成方式，不能据此声称人类识别率≥95%。

固定数据与预览在 `training/data/degradation81-calibration-v1/`（生成物按现有规则不进 Git）。
该目录的 `manifest.json`、`audit.json` 在本记录目录留存快照，以便追溯。
上述数据审计在 2026-10-20 完成；2026-10-27 的模型准确率及混淆关系见本记录顶部的新结果。人工可读范围尚未实测。

## 在 Colab 完成真实扫描

上传 [`hanzi81_degradation_phase1.ipynb`](../../hanzi81_degradation_phase1.ipynb)，按顺序运行两格即可。
无需上传新数据包、模型或辅助源码；Notebook 从原 Drive 数据 ZIP 中只解出 `val_graph.npz` 和三份元数据。

1. 第①格挂载 Drive，确认原模型身份，固定 Pillow=12.3.0，生成并校验与本地一致的像素。
2. 第②格运行25档推理，先在 `/content` 完成报告，再保存到新 Drive 时间戳目录，自动下载结果 ZIP。

权重：`MyDrive/hanzi81/runs/baseline81_s42_v1/best.pt`。
原数据包：`MyDrive/hanzi81/hanzi81_colab_bundle.zip`，或原会话已保存的 `packages/<原包SHA256>.zip`。
报告：`MyDrive/hanzi81/analyses/baseline81_s42_v1/07_degradation_phase1/<北京时间戳>/`。

报告包含 `summary.json`、`metrics.json`、`summary.csv`、`pairs.csv`、`predictions.csv`、`confusion.npz`、
`accuracy_curves.png`、`report.html`、标定协议和审计快照、评估源码及结果ZIP。
每档54对的双向误认分别统计；支持规则沿用“每个源类别最常误认目标至少5次”。
不能跨25档累加同一来源的派生错误，制造虚假的支持数。

## 验收与下一阶段门槛

第一阶段完整验收：原模型身份匹配，25档共32,400次预测齐全，25个混淆矩阵各合计1,296，
清晰对照应复现100%，结果保存到Drive。清晰对照异常时先排查预处理、模型与推理环境。
只有真实扫描报告完整时，状态才改为第一阶段完成。

下一阶段先人工盲认，再在固定可读范围内重训普通交叉熵基线。此前讨论的“人工≥95%、
重训后仍有≥1%错误并存在≥5次支持的形近误认”是拟定立项门槛，尚未测得。
若可读范围内重训后仍接近满分，停止单纯加强模糊，转向字表或字体覆盖；不可辨档只作极限压力测试。

## 实现与验证入口

- [`training/degradation.py`](../../../training/degradation.py)：生成、像素审计、54对预览、冻结模型推理与报告。
- [`training/test_degradation.py`](../../../training/test_degradation.py)：针对性测试，不运行原项目全套训练测试。
- [`colab/make_degradation_notebook.py`](../../make_degradation_notebook.py)：将冻结模型构造及新评估代码嵌入两格Notebook。

```bash
python -m training.degradation generate --output training/data/degradation81-calibration-v1 --font pilot/fonts/SourceHanSansSC-Regular.otf
python -m colab.make_degradation_notebook
python -m unittest training.test_degradation -v
```

生成器拒绝覆盖已有数据目录；验证已有交付时直接运行测试，无需重新生成整批数据。
本地推理链测试使用三类、150张的合成夹具和已知输出的真实ResNet，验证统计与报告，不产生研究准确率。
本地运行环境及实际验证结果见本目录 `implementation_status.json`。
