# 扫描 PDF：PaddleOCR、Edge 与 81 字模型的第一阶段实验

状态：**方案与工具已准备，真实扫描数据和正式对照实验尚未完成**。不使用原基线 test，不修改原权重，不训练模型。

本机联通检查：PaddleOCR 3.4.1 / PaddlePaddle 3.2.0 与官方 PP-OCRv5 推理模型已在合成图片上跑通；PyMuPDF 1.26.7 已验证纯图渲染和文字层拒绝；原 `best.pt` 与归档指纹一致，人工构造的 `士`→`土` 单字样例跑通模型和候选生成；配对工作流针对性测试通过。上述均是工具检查，**不是扫描文档实验结果**；`士` 是人为构造的输入，不是 PaddleOCR 在该图上的实测错误。

## 问题与决策

在中文印刷扫描页面上，81 字模型能否纠正 PaddleOCR 的形近字错误，而不产生更多错误？Edge 的扫描 PDF 识别作为独立外部效果对照。

| 组 | 输入和输出 | 目的 |
|---|---|---|
| A | PaddleOCR `PP-OCRv5_server_rec` 识别人工裁好的行 | 基线 |
| B | A 的文本 + 人工字框 + 真值辅助位置对齐 + 单字模型复核 | 定位理想时的诊断上限 |
| D | Edge 打开同一纯图 PDF，复制各页识别文字 | 独立外部对照 |

B 用真实文本的编辑距离对齐，在插入、删除或重复字附近可能有多种同样好的对齐；它既是理想化诊断，也可能因歧义定位错误。不可称为自动系统表现。D 是整页识别，A/B 是人工行裁图识别；不据 D 与 A/B 的差异断言 OCR 引擎本身的优劣。C（自动分割）仅在 B 显示净收益后另行设计，不属于本次代码。

原模型为 81 类、固定字体合成数据训练。开发集 100% 不证明真实文档有效。`best.pt` 应为原第 38 轮 baseline 权重；代码核验权重与[原回传记录](../baseline81_s42_v1/reported_results.json)中的 SHA-256、训练数据 manifest 和标签文件。原 [81 字实验记录](../baseline81_s42_v1/README.md)描述其适用边界。

## 数据与标注

收集约 20 个来源页面、约 200 行及 4,000 字符，按来源约 1:1 分为开发和测试；同一文档/字体来源的相关页面不跨集合。记录来源、字体、扫描条件；这只是探索规模，不是统计功效保证。包含目标字、普通字和字表外字，不按 OCR 错误挑选主测试集。只选可完整转录的单栏印刷页面，页眉、页脚等可见文字也必须纳入；有无法确认真值的文字时，在**看 OCR 结果之前排除整页**并登记，不得按结果筛选或只排除其中一行。这样 Edge 的整页结果与 A/B 的全部行对应相同评价范围。标注第二人或复核者对原图复核。

PDF 必须为纯图扫描版，确认无既有文字层：下面 `prepare-pdf` 用 PyMuPDF 查找可提取文本，发现则报错；它不能保证发现所有异常文字层，仍需检查源文件与渲染页。不要把 Edge 识别后保存的新 PDF 作为源文件。

先从渲染页人工裁出文字行，保持原分辨率和阅读顺序。对每一行的**全部字符**（含标点及字表外字）手工标框，坐标相对于行图的左上角，格式 `[左, 上, 右, 下]`，右、下边界不包含在框内；保留完整笔画及适量背景。无法标框的位置写 `null` 并登记，不因模型识别是否正确决定是否标框。尽量减少 `null`：人工字框不完整会降低实际覆盖，使 B 不再代表理想定位。

为每页准备一个 UTF-8 文本文件，原样保存 Edge 复制的文字。复制失败时创建空文件并在清单中标记 `failed`，不跳过页面。文字真值不包含空白，评价时只移除预测中的所有空白；标点和汉字变体不替换。逐页行顺序事先确定，拼接后评估；段落和换行差异不计入 CER，漏字、多字和页面失败计入。将 Edge 版本、操作系统版本、PDF 来源、页码、裁图人员和日期记入清单的 `notes`。敏感或未获授权的材料不要提交或上传。

## 环境和材料

准备一个有 Python、PyTorch、Torchvision、NumPy、Pillow、PyMuPDF、PaddlePaddle 和 PaddleOCR 的运行环境。项目训练环境与 Paddle 组合需事先确认可用；不要改变原训练环境。建议参考 [PaddleOCR 3.4.1 安装说明](https://www.paddleocr.ai/v3.4.1/en/version3.x/installation.html)，CPU 选 PaddlePaddle 3.2.0，并安装 `paddleocr==3.4.1`；按实际 Python/平台选可用的 torch、torchvision、PyMuPDF 与 Pillow。环境版本写入实验记录，脚本保存 Paddle 和 torch 版本。首次下载模型和包需要网络。

例如，在 Linux 中用合适的 Python 3.13 新建独立环境（不要在原 `training/.venv` 中安装 Paddle）：

```bash
python3 -m venv work/venv
work/venv/bin/python -m pip install torch==2.11.0 torchvision==0.26.0 Pillow PyMuPDF==1.26.7 numpy==2.2.6
work/venv/bin/python -m pip install paddlepaddle==3.2.0 paddleocr==3.4.1
work/venv/bin/python -c 'import torch, torchvision, fitz, paddle, paddleocr; print(torch.__version__, torchvision.__version__, fitz.VersionBind, paddle.__version__, paddleocr.__version__)'
```

随后用 `work/venv/bin/python` 替代下方所有 `python`。也可以分开使用 Paddle 环境和项目训练环境：`prepare-pdf`、`validate` 和 `paddle` 使用含对应依赖的环境，`candidates` 使用原模型的 PyTorch 环境；`tune` 和 `report` 只需 Python 标准库。开发与测试各自的对应阶段必须使用同版本环境。

本机已在 `training/.verification/ocr-paddle-env/` 准备独立的 Windows OCR 验证环境，可使用 `& '.\training\.verification\ocr-paddle-env\Scripts\python.exe'` 执行 `prepare-pdf`、`validate` 和 `paddle`。该环境是本地缓存，不属于 Git 交付。

若换机器或需要新建隔离环境，Windows PowerShell 的分环境做法如下（首次安装需要下载依赖）：

```powershell
& 'D:\Tool\Python\Python313\python.exe' -m venv work\ocr-env
& '.\work\ocr-env\Scripts\python.exe' -m pip install paddlepaddle==3.2.0 paddleocr==3.4.1 PyMuPDF==1.26.7 Pillow numpy==2.2.6
```

使用下面的命令时，将 `prepare-pdf`、`validate`、`paddle` 的 `python` 替换为 `& '.\work\ocr-env\Scripts\python.exe'`，将 `ocr_label`、`candidates` 的 `python` 替换为 `& '.\training\.venv\Scripts\python.exe'`；`assemble`、`tune`、`report` 可使用任一环境。PowerShell 命令中的路径和参数其余部分保持不变。两个环境都在同一个仓库根目录运行，不能同时写同一个输出文件。

从 [PaddleOCR 3.4.1 文字识别模型表](https://www.paddleocr.ai/v3.4.1/en/version3.x/module_usage/text_recognition.html)下载 `PP-OCRv5_server_rec` 的 **Inference Model**（不是 Pretrained Model），解压到固定目录。以下命令以项目根目录为工作目录、`python` 为实验环境解释器、`work/` 为本地数据目录；按实际路径替换。模型目录应直接包含推理配置与权重文件。脚本对目录内容计算指纹，开发与测试必须一致。

本机验证下载的推理模型位于 `training/.verification/paddle-model/PP-OCRv5_server_rec_infer/`，可直接作为 `--model-dir` 使用；该缓存不属于 Git 交付，换机器时按上面的官方链接重新下载并固定。

```text
work/
  dev.json                    # 开发清单
  test.json                   # 测试清单
  lines/xxx.png               # 人工裁剪的行图
  annotations/xxx.json        # 字框标注工具逐行导出
  edge/xxx.txt                # 每页 Edge 原始复制文本
  model/PP-OCRv5_server_rec/  # 官方推理模型
  out/                        # 新的输出文件
```

`work/` 应放在实验环境可写、不会被公开提交的目录；不要把它视作本仓库 Git 跟踪路径。模板见 [manifest.example.json](manifest.example.json)，它仅示意格式，不是实验数据。`truth` 不含空格，`boxes` 长度等于 Python 字符数；如存在组合字符或复杂文字，先排除并记录。清单按页和行的数组顺序确定阅读顺序。页 ID 全局唯一，source_id 表示真实来源，两个清单中必须保持一致。

## 实际操作顺序

以下命令从仓库根目录运行；`work` 可替换成绝对路径。若本地 `colab/best.pt` 可用，下面可以用该路径；仍由脚本检查 epoch、模式、manifest 和标签对应关系。若不匹配，请使用原 Drive 中正确的权重，不要改代码强行绕过。`training/data/baseline81-v2/` 只需原始 `manifest.json` 与 `labels.json`，不会读取原 test 数据。

1. 渲染每份 PDF 到**新的**目录（一个 PDF 对应一个输出目录），检查 `pdf_info.json` 和所有页面；然后人工裁行、标字框、转录、复制 Edge 文本并制作两份清单：

   ```bash
   python -m training.ocr_pilot prepare-pdf work/source.pdf work/render/source --dpi 200
   ```

   在 Edge 打开相同原始 PDF，逐页选择并复制完整页面的识别结果至对应 `edge/*.txt`，确认每个文件只含本页文字；核对没有漏页，但不要根据真值修正 OCR 文本。页面无法识别就保留空文件并标 `failed`。Edge 的识别与界面版本可能变化，必须在 notes 记录。

   可使用随附的图形标注器逐行框字（需要桌面环境、Python Tk 和 Pillow）：

   ```bash
   python -m training.ocr_label work/lines/page_001_line_001.png --id line_001 --truth '土地。' --manifest-dir work --output work/annotations/page_001_line_001.json --zoom 2
   ```

   将 `土地。` 换成该行人工转录；在窗口中按提示的顺序拖出每字完整框，使用“撤销”或“跳过”，完成后点击“保存”。`--zoom` 仅控制显示大小；过宽时可设为 `1` 或 `0.5`。关闭未保存的窗口不会生成标注。输出文件不可覆盖：要重标请另存新文件并在清单中引用。逐行标注后，参照 [draft.example.json](draft.example.json) 创建 `work/dev_draft.json` 和 `work/test_draft.json`，按阅读顺序列出每页的标注文件；`edge_text` 和 `lines` 路径相对于草稿所在目录。用以下命令生成与草稿同目录的最终清单：

   ```bash
   python -m training.ocr_pilot assemble work/dev_draft.json work/dev.json
   python -m training.ocr_pilot assemble work/test_draft.json work/test.json
   ```

   也可直接按 [manifest.example.json](manifest.example.json) 编辑最终清单。每页行图必须与其原始 PDF 页面相对应；标注器只记录已有行图的路径，不自动确定来源。

2. 同时检查两个清单、源分组、行图尺寸、字框边界和 Edge 文件：

   ```bash
   python -m training.ocr_pilot validate work/dev.json work/test.json
   ```

   校验通过后冻结数据和人工标注。脚本不自动验证 PDF、图片、Edge 版本或人工转录之间的对应关系；这些需人工核对。

3. **只处理开发集**，选择阈值。每个输出必须为新文件，避免覆盖旧结果：

   ```bash
   python -m training.ocr_pilot paddle work/dev.json work/out/dev_a.json --model-dir work/model/PP-OCRv5_server_rec --device cpu
   python -m training.ocr_pilot candidates work/dev.json work/out/dev_a.json training/data/baseline81-v2 colab/best.pt work/out/dev_candidates.json
   python -m training.ocr_pilot tune work/dev.json work/out/dev_a.json work/out/dev_candidates.json work/out/policy.json
   python -m training.ocr_pilot report work/dev.json work/out/dev_a.json work/out/dev_candidates.json work/out/policy.json work/out/dev_report.json
   ```

   `policy.json` 在固定阈值 `[不替换, 0.99, 0.95, 0.9, 0.8, 0.7, 0.5]` 中选开发集编辑距离最小者；并列优先较少替换。阈值是单字模型 softmax 分数，不是已校准概率。只允许预设的 54 个形近对互相替换；不自动修复漏字或多字。处理方式：手工字框裁图按最长边缩放至 48 像素以内，居中贴在 64×64 白底上，然后 `x / 127.5 - 1`。裁图尺度与训练渲染方式不完全相同，需在开发集查看实际裁图并据此解释结果；若要变更，必须在接触测试结果前冻结新协议。

4. 冻结开发选择，再运行测试集，**不得在查看测试结果后回头选阈值**：

   ```bash
   python -m training.ocr_pilot paddle work/test.json work/out/test_a.json --model-dir work/model/PP-OCRv5_server_rec --device cpu
   python -m training.ocr_pilot candidates work/test.json work/out/test_a.json training/data/baseline81-v2 colab/best.pt work/out/test_candidates.json
   python -m training.ocr_pilot report work/test.json work/out/test_a.json work/out/test_candidates.json work/out/policy.json work/out/test_report.json
   ```

   脚本核验清单和其引用的行图、Edge 文本没有在各阶段间变化；也要求开发/测试的 Paddle 版本、模型、设备、单字权重和标签一致。代码不自动阻止反复查看测试集，因此需要人为执行冻结协议。不要把测试输出用于重新拟合阈值。

## 报告和停止条件

`test_report.json` 的 `cer` 给出 A、B、D 的整页 CER；`B_minus_A` 为 B 减 A，负值表示 B 更好。CER = 各页编辑距离之和 / 真值字符总数。`pages` 给逐页错误和 Edge 状态；`coverage` 给输出目标字数、真值辅助对齐数、缺框数和实际复核数；`changes` 是所有实际替换和对应真值。`reference_group_errors_excluding_insertions` 按真值属于 81 字或字表外统计替换/漏字，**不含多字**，不能当作该组 CER。

`paired_bootstrap_95` 将文档来源作为重采样单位、固定种子做 2,000 次配对 bootstrap，给出 B−A、D−A 和 D−B 的差值区间；D 相关差值仍受输入流程差异影响。来源太少时区间不可靠；只有一个来源时输出 `null`。报告目标字覆盖、逐页结果、纠错/误改/错改错和字表外真值被替换数。检查字框缺失和对齐歧义，再解读收益。

预注册判定：B 在测试集相对 A 的 CER 至少下降 **1 个百分点**，配对区间上界小于 0，且审查误改与各来源结果，才支持进入自动定位实验；这个门槛是目标，不是已测结果。若 B 无净收益，停止开发分割器，先解决真实裁图/字体与字表外问题；若相关错误太少或区间过宽，增加独立来源，不作肯定或否定结论。Edge 结果用于外部对照，不作为判断 B 是否有效的替代证据。
