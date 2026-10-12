# 字体样张核验

本目录保存 81 字、54 个重点字对的首版核验材料及样张生成工具。`samples/v1` 仅用于检查固定字体、图像条件和标签，不是训练或测试数据。本目录另含 PR #1 的独立小规模基线工具，见下文；它与当前 Colab 基线使用不同的数据版本和输出目录。

## 从哪里查看

- [交互核验页面](./samples/v1/index.html)：清晰总览、字对对照、全部字符的位移检查，支持原尺寸和放大显示。
- [清晰字表总览](./samples/v1/overview_clear.png)、[sigma=1 字表总览](./samples/v1/overview_sigma1.png)。
- [样张核验记录](./样张核验记录.md)：自动检查结果、视觉检查范围及仍需区分的结论。
- [参数与版本](./samples/v1/config.json)、[自动核验详情](./samples/v1/audit.json)、[逐图清单](./samples/v1/manifest.jsonl)。

页面直接在浏览器打开，无需联网或启动服务。字对页面中的人工标记保存在当前浏览器；使用“导出”按钮保存核验记录，跨浏览器不会自动同步。自动检查结果与人工标记是两种不同证据。

## 本次渲染规则

以 `characters.json` 为标签来源，使用同一个思源黑体 SC Regular 文件，不使用字体回退。所有字符在四倍画布上采用共同字号与共同基线；共同定位基于全部 81 字相对于同一基线的包围盒并集，不逐字紧框裁剪或单独拉伸。

1. 输出为 64×64、8 位灰度、白底黑字。
2. 在 256×256 尺度绘制，共同字号为 192 px，即输出尺度 48 px。若字表改变导致联合包围盒超出 192×192，则脚本统一减小字号并要求另用输出目录。
3. 使用固定 Lanczos 下采样；之后应用 `sigma=0、0.5、1.0` 的 Pillow GaussianBlur。模糊前四周补 16 px 白边，处理后裁回原画布。
4. 检查位置为居中，以及横纵均为 ±3 px 的四个角向组合：每字 3×5=15 张，共 1,215 张。这只覆盖核验位置，不代表正式 ±3 px、0.25 px 步长的完整采样。
5. PNG 只包含汉字图像；标签、边框和注释只出现在总览页和对照页。放大图使用近邻复制像素，没有重新渲染字符或补造细节。

81 字、54 对仍是人工候选关系。共享构件不代表像素完全相同；自动检查通过也不等于所有字对都易混淆或方法已经有效。

## 文件与来源

| 文件 | 用途 |
|---|---|
| 核验字表.md、characters.json | 用户提供的核验字表，生成器只读 |
| prepare_font.py | 下载官方字体及许可证；已有资源会核对哈希后复用 |
| fonts/ | 字体、许可证及固定来源元数据 |
| render_samples.py | 生成单字图、自动核验、对照图及离线页面 |
| gallery_template.html | 离线核验页面模板 |
| samples/v1/ | 本次输出；包含全部 PNG 和生成记录 |
| requirements.txt | 此次实际使用的 Python 包版本 |

字体来自 [Adobe 官方 Source Han Sans 仓库](https://github.com/adobe-fonts/source-han-sans)，固定源提交与 SHA-256 见 [fonts/source.json](./fonts/source.json)，许可证随文件保存在 [fonts/LICENSE.txt](./fonts/LICENSE.txt)。字体版本由下载文件的 name 表读取，而非仅凭文件名推断。

有关 API 含义见 [Pillow 字体基线与包围盒](https://pillow.readthedocs.io/en/stable/reference/ImageFont.html#PIL.ImageFont.FreeTypeFont.getbbox)和 [GaussianBlur 参数](https://pillow.readthedocs.io/en/stable/reference/ImageFilter.html#PIL.ImageFilter.GaussianBlur)；本次实际软件版本记录在 config.json。

## 复现

在 `pilot` 目录内运行。首次准备可使用 Python 3.13 建立独立虚拟环境，再安装 `requirements.txt`。本机 `.venv` 使用已安装的 Pillow，并在项目环境内安装 fonttools；包版本已记录，未修改系统 Python 包。

```powershell
# 已配置好的本机环境
.\.venv\Scripts\python.exe prepare_font.py
.\.venv\Scripts\python.exe render_samples.py
```

首次下载字体需要网络；已有字体和许可证的哈希正确时，生成样张可离线完成。用新环境复现时先安装依赖：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe prepare_font.py
.\.venv\Scripts\python.exe render_samples.py
```

重新生成样张会覆盖样张生成器管理的图片、页面和 JSON 记录，不修改输入字表及独立的核验记录文档。若字表、字体、尺寸、位置、模糊级别、字号或基线改变，使用新输出目录，例如 `render_samples.py --output samples/v2`。不能把这些核验图片直接称作独立测试集。

## PR #1 独立基线：修复后运行方法

这些命令从仓库根目录执行，使用单独的本地 Python 环境。`requirements-training.txt` 补充训练依赖；torch 与 torchvision 应使用当前平台相互兼容的版本。本流程默认 CPU，不修改当前 Colab 环境、上传包、Drive 数据或训练记录。

```text
python -m pip install -r pilot/requirements-training.txt
python pilot/build_dataset.py
python pilot/train_baseline_a.py --seed 42
python pilot/eval_confusion.py
```

- 新数据写入 `pilot/dataset_81_v2/`。81 类各有 train 20、val_graph 8、val_select 6、test 6 张，总计 3,240 张；图像为 64×64 灰度。它不替代 Colab 使用的 46,656 张数据。
- 所有字共用同一份条件表：偏移范围 ±3 px、步长 0.25 px，sigma 为 0/0.5/1。每类三档模糊数量分别为 train 7/7/6、val_graph 3/3/2、val_select 2/2/2、test 2/2/2。生成全部候选像素、全局去重后才划分；如果某条件发生像素重复，就为所有类别一起拒绝该条件。小样本条件表仍不代表完整条件网格或正式统计功效。
- `meta/dataset_meta.json` 保存实际渲染参数及解码像素 SHA-256，不再对参数二次取整。`meta/manifest.json` 记录字体、字表、生成器、软件版本和分布。加载器验证元数据及图片哈希，旧数据需重新生成。
- 新训练写入 `pilot/checkpoints_a_v2/`，保存配置、完整精度日志及最佳检查点。Python、NumPy、PyTorch 和数据加载器都固定种子；按 val_select 准确率选择模型，同分时比较交叉熵。先保存再早停，首轮 100% 或全程 0% 都有检查点。重复性仅限相同软件、设备与执行设置；这里没有实现断点续训。
- 数据、训练和评估的输出目录已存在时会拒绝覆盖。另一次实验需指定新的 `--output` / `--save-dir`，并把相应 `--data-root` / `--checkpoint` 显式传给后续步骤。训练与评估检查数据版本匹配，不会自动沿用旧检查点。
- 混淆评估只读取 val_graph，结果写入运行目录下的 `val_graph/`。无误认的类别以 `null` 表示无混淆目标；`--plot` 可选生成图表，需自行在该独立环境安装 matplotlib。test 不用于模型选择或混淆挖掘。

原 PR 的 `confuse_pairs.json` 已改名为 `confuse_pairs_legacy_unverified.json`，内容保留，仅作历史记录。旧流程存在跨分区重复、最佳模型漏存和训练种子未固定的问题，因此旧准确率与空混淆集合不能作为修复后实验的结果，也不能推出“不存在统计上稳定的混淆”。应完成新的基线运行后再分析，不通过削弱模型或调整阈值制造收益。

## 修复验证

```text
python -m unittest discover -s pilot/tests -p "test_*.py" -v
python pilot/tests/verify_pipeline.py
```

第一条包含真实渲染和两次同种子生成，逐张核验标签、唯一像素、分区、条件分布和从参数重建；还覆盖首次/后续达到 100%、全程 0%、同种子重复训练、已有输出保护、数据版本不匹配及无误认记录。生成测试只在 `pilot/.verification/` 临时目录中读写。

第二条在已生成的 `dataset_81_v2` 上运行真实 ResNet18：每字取 2 张训练图、2 张 val_select 图，CPU 跑 1 轮，重载检查点，再检查全部 648 张 val_graph 的评估流程。输出保存在 `pilot/.verification/pipeline/`，不读取 test 图片。该检查验证流程，不是 30 轮基线结果；再次运行可指定新的 `--output`。

2026-10-12 本地验证通过：3 项数据测试、7 项训练/评估回归测试，以及上述真实模型流程检查。3,240 张图片全部唯一、标签与分区数量正确、参数重建逐像素一致；相同种子两次生成的 PNG 和元数据字节一致。首轮/后续 100%、全程 0% 均保存可加载检查点；相同种子训练的日志和权重一致，改变种子会改变权重。验证环境为 Python 3.13.15；渲染使用 Pillow 12.3.0、fonttools 4.66.1、FreeType 2.14.3；训练使用 PyTorch 2.14.1+cpu、NumPy 2.5.2。本次未运行修复版 30 轮正式基线，也未在 GPU 上验证；数据审计会检查 test 的像素完整性，但没有对 test 执行模型推理。
