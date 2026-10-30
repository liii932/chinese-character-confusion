# 第二轮：完整 1,140 字数据与 10 轮 Colab 入口

日期：2026-10-29（Asia/Shanghai）。用户要求按设计中的全部字对／字组先生成第二轮训练图片。
本轮采用完整候选池，沿用首轮生成机制；1,140 类训练已收到用户回传结果，PaddleOCR 扩类对照未开展。

状态：656,640 张图片已生成，独立重新解码四个集合后的全量像素审计通过。2026-10-29 用户回传已完成 10 轮训练，
同一第 10 轮检查点在 val_select 与 val_graph 各 109,440 张上的准确率均为 100%，三个 sigma 分组也均为 100%。
原始回传见 [reported_results_2026-10-29.json](reported_results_2026-10-29.json)；尚未独立读取 Drive 权重或完整 history，test 未评价。

交付数据包：`colab/dist/hanzi1140_round2_data.zip`；训练入口为 [`hanzi1140_round2_e10.ipynb`](../../hanzi1140_round2_e10.ipynb)，直接复用原数据 ZIP。
本机可直接打开 `training/data/round2-1140-v1/preview/index.html` 浏览全部字组。

## 范围与关系

- 原表 443 个条目、442 个不同字组、1,190 次字符出现，去重为 1,140 类。重叠字符不重复生成或加权。
- [完整字表](../../../training/characters1140.json)保留原 81 字的类别 ID 0～80，新增 1,059 字按原表首次出现顺序追加为 81～1139。
- `source_groups` 保留全部原始条目、来源行号和类别；`candidate_pairs` 登记原表双字条目的 251 条无向候选关系，重复来源合并。
- `pairs` 单独保留首轮已核验的 54 对。多字组不自动展开为全部两两关系，不做传递扩展；尚未完成新字对的差异标注或主要评价关系冻结。

## 固定数据协议

| 项目 | 本轮值 |
|---|---|
| 数据目录 | `training/data/round2-1140-v1/` |
| 数据种子 | 20261029 |
| 字体 | 原已固定的 Source Han Sans SC Regular 2.005 OTF，沿用原字体文件及许可 |
| 字体覆盖 | 1,140 字全部有有效字形 |
| 图像 | 64×64、uint8 灰度、白底黑字；训练像素无损保存在 NPZ 中 |
| 渲染 | 256×256 画布、192 px 统一字号，Lanczos 下采样一次；不逐字裁框缩放 |
| 全字表包围盒并集 | `[0, -165, 192, 20]`，相对于左基线锚点的高分辨率坐标 |
| 全字表共同基线 | `[32, 201]`；由完整字表重新计算，首轮为 `[32, 200]` |
| 位置 | 横纵 ±3 px，步长 0.25 px |
| 模糊 | sigma=0、0.5、1.0，每档等量；模糊前补16 px白边，处理后裁回 |
| 归一化 | `uint8 / 127.5 - 1`，无在线增强 |
| 划分 | 全字表共享条件，先做像素去重，再分配四个集合；每集合／sigma／方向偏移均值绝对值≤0.25 px，两端覆盖至少±2.5 px |

| 集合 | 每字图数 | 每字每档 sigma | 全表图数 |
|---|---:|---:|---:|
| train | 192 | 64 | 218,880 |
| val_graph | 96 | 32 | 109,440 |
| val_select | 96 | 32 | 109,440 |
| test | 192 | 64 | 218,880 |
| 总计 | 576 | 192 | 656,640 |

每份 NPZ 包含 `images`、`labels`、`condition_ids`，兼容已有训练读取接口。
`labels.json` 保存字符与关系，`conditions.json` 保存各集合条件，`manifest.json` 保存配置、运行环境和文件指纹。
`audit.json` 来自单独重新解码四份 NPZ 的离线检查；不在训练启动时重复执行全量像素审计。
`preview/index.html` 按全部 443 个来源字组浏览实际训练图的 sigma=0／1 对照；12 张 overview 图覆盖所有类别。

当前 test 只做图片与元数据质量检查，没有模型评价。原退化标定集和旧 test 不被并入第二轮训练。
四集合互斥保证针对本轮数据；跨轮使用相同字模与有限网格，不保证新旧图片互斥。
第二轮应新建 1,140 类、随机初始化的基线 run，不能把旧 81 类模型直接续训或在新 test 上视为独立评估。
同一字模的亚像素变体属于固定合成分布，不代表独立真实字形、跨字体或真实拍照 OCR 样本。

## 生成与验收

```bash
python -m unittest training.test_round2 -v
python -m training.prepare_round2
python -m training.audit_round2
```

普通运行使用安装了 NumPy、Pillow=12.3.0、fonttools=4.66.1 的 Python 环境。
生成器拒绝混入已有不同配置或未完成目录；`manifest.json` 是完整生成的标记。
重新生成时使用新的空输出目录，例如 `python -m training.prepare_round2 --output /path/to/new-data`。
实际依赖版本、FreeType 版本和数据指纹以 manifest 为准。

本轮实测环境：Python 3.14.4、NumPy 2.5.4、Pillow 12.3.0、fonttools 4.66.1、FreeType 2.14.3。
两项字表与关系测试通过；四集合解码审计确认 656,640 个唯一像素哈希，空图、非白边界、
像素重复和跨集条件重复均为 0，标签／条件索引、配额与位置均衡符合协议，文件指纹全部一致。
已检查总览第 1、12 页的实际渲染，未发现明显裁剪；这不是人工盲认可读性测量。
本目录留存生成 manifest、audit 和交付包 delivery 元数据快照。

## Colab 训练协议与操作

将用户“训练10层”按 **10 个 epoch（轮次）**配置，架构继续沿用首轮的 ResNet18。
Notebook 内嵌原 `training/train.py`、`io_utils.py` 和包入口，训练核心保持首轮实现；具体版本见 `notebook_manifest.json`。

| 项目 | 第二轮配置 |
|---|---|
| 分类头与初始化 | 1,140 类，全部随机初始化 |
| 完整基线预算 | 10 epoch；`--epochs 10`，分段运行时仍保留该总预算 |
| 优化 | 普通交叉熵、AdamW、batch=64、lr=0.001、weight_decay=0.0001、seed=42 |
| 学习率 | 余弦 T_max=10，终点为初始学习率的 1/100 |
| 精度 | CUDA AMP 训练、fp32 验证 |
| 选择与保存 | val_select macro accuracy 最高；并列取 CE 更低者；逐轮保存 best/last.pt、配置、完整恢复状态及 history |
| 独立短跑 | 每类 2 张训练及 2 张验证图、1 epoch，另存 `<RUN_NAME>_smoke`；完整基线重新初始化 |
| 结果 | 曲线、val_select 与 val_graph 的总体／sigma 准确率、混淆矩阵、原 54 对统计、逐图预测；不执行 test 推理 |

1. 在 `MyDrive/hanzi1140_round2/` 上传原数据包 `hanzi1140_round2_data.zip`。
2. 上传新 Notebook 到 Colab 并保存 Drive 副本，选择 GPU，顺序运行六格。
3. 第⑤格默认完成全部 10 轮。`EPOCHS_THIS_SESSION=2` 表示本次最多追加两轮；断线后重跑准备格再运行本格，自动恢复。
4. 结果保存在 `MyDrive/hanzi1140_round2/runs/baseline1140_e10_s42_v1/`。

第⑥格保存 `learning_curves.png`、`notebook_summary.json` 和两个 `evaluation_*` 目录。
只在完整 10 轮后生成开发报告，已完成的同一 best.pt 报告会复用；10 轮是本次预算，不预先声称收敛或某个准确率。
上传的数据 ZIP 保持原冻结版本；本次新增的 Notebook 配置与验证记录单独保存。

维护入口为 `colab/round2_cells/*.py` 与 `colab/make_round2_notebook.py`。
`python -m unittest colab.test_round2_notebook -v` 的 5 项离线检查通过，覆盖：选择性数据恢复与源码冻结、损坏上传包拦截、
两轮后恢复仍传入十轮总预算、开发报告的完整训练门槛与检查点缓存、Notebook 嵌入源码与空输出。
这些检查不执行 GPU 训练，不产生模型准确率；后续 Colab 实际运行的用户回传结果见本记录顶部。
