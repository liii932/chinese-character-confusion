"""Clone the completed baseline notebook into a separate R16 experiment."""
import copy
import json
from pathlib import Path
import re
from training.io_utils import digest
ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'training/data/lowres16-1140-v1'
RECORD=ROOT/'colab/experiments/lowres16_1140_v1'

def generate():
    original=json.loads((ROOT/'colab/hanzi1140_round2_e10.ipynb').read_text(encoding='utf-8'))
    old=[c['source'] for c in original['cells'] if c['cell_type']=='code']
    delivery=json.loads((RECORD/'delivery.json').read_text(encoding='utf-8'))
    setup=old[1].replace('hanzi1140_round2','hanzi1140_r16').replace('round2-1140-v1','lowres16-1140-v1')
    setup=re.sub(r'(?m)^DATA_ARCHIVE_SHA256 = .*', 'DATA_ARCHIVE_SHA256 = '+repr(delivery['sha256']),setup)
    setup=re.sub(r'(?m)^DATA_MANIFEST_SHA256 = .*', 'DATA_MANIFEST_SHA256 = '+repr(digest(DATA/'manifest.json')),setup)
    setup="import torch\nif not torch.cuda.is_available():\n    raise RuntimeError('请选择Colab GPU运行时。')\nprint('GPU:', torch.cuda.get_device_name(0), 'PyTorch:', torch.__version__)\n\n"+setup
    helper=old[3].split('\nsmoke_dir =')[0].replace('baseline1140_e10_s42_v1','baseline1140_r16_e10_s42_v1')
    training=helper+'\n\n'+old[4]
    cells=[]
    def md(s):cells.append({'cell_type':'markdown','metadata':{},'source':s+'\n'})
    def code(s):
        compile(s,'R16-cell','exec')
        cells.append({'cell_type':'code','metadata':{},'source':s,'outputs':[],'execution_count':None})
    md('''# R=16、sigma=0：1,140字同分布基线，10轮

本Notebook复用已完成的第二轮训练核心，另建R16数据及run；模型仍为ResNet18，全部随机初始化。
训练、val_select、val_graph及封存test统一使用：64像素清晰字图 → Lanczos缩至16 → bicubic恢复64。
字号、字体、位置网格、每类配额沿用原设计；sigma固定为0。用户已人工确认可辨认，不再执行盲认或评分。

1. 在Drive“我的云端硬盘”建立 `hanzi1140_r16`，上传 `hanzi1140_r16_data.zip`。
2. 上传本Notebook到Colab并保存副本，选择GPU，顺序运行下面三个代码单元。
3. 默认完整训练10轮，逐轮保存。断线后重新运行准备格及训练格，恢复最近完整epoch。

同一run只在一个会话中写入。数据与源码指纹检查用于保持训练输入一致，不进行新增可读性检测。''')
    md('''## ① 准备训练数据与代码

挂载Drive，复制并解出固定数据文件到/content。训练核心已内嵌，不需要另外上传脚本。
Drive项目目录为 `MyDrive/hanzi1140_r16`，不使用旧81字或旧1140字检查点。
本格只输出设备信息并准备文件，不运行模型短跑或盲认。''')
    code(setup)
    md('''## ② 训练10轮

沿用普通交叉熵、AdamW、batch64、lr=0.001、weight_decay=0.0001、seed42、CUDA AMP训练和fp32验证。
余弦学习率总预算固定10轮。`EPOCHS_THIS_SESSION=None` 完成剩余全部轮次；改为2表示本次最多追加两轮，总预算不变。
选择best.pt使用val_select宏准确率最高，并列取CE更低者；last.pt逐轮保存完整续训状态。
run目录：`MyDrive/hanzi1140_r16/runs/baseline1140_r16_e10_s42_v1/`。''')
    code(training)
    md('''## ③ 曲线与两个开发报告

保存训练/验证损失与准确率曲线。完整10轮后分别评价val_select和val_graph，保存逐图预测及1140×1140混淆矩阵。
报告中的sigma仅有0；全部图片的中间分辨率均为16。保留原54对误认统计，并列出全字表错误最多的类别。
本Notebook不评价test。10轮是本次预算，不提前认定收敛或准确率；根据实际曲线与错误分布决定后续阶段。
确认Drive文件保存完成后，断开连接并删除运行时。''')
    code(old[5])
    nb=copy.deepcopy(original);nb['cells']=cells
    nb['metadata']['colab']['name']='hanzi1140_r16_e10.ipynb'
    nb['metadata']['round2_protocol'].update({'resolution':16,'sigma':0,'data_archive_sha256':delivery['sha256'],'data_manifest_sha256':digest(DATA/'manifest.json'),'readability_basis':'user confirmation; no further human tests'})
    for i,c in enumerate(cells):c['id']=f'r16-{i:02d}'
    target=ROOT/'colab/hanzi1140_r16_e10.ipynb';target.write_text(json.dumps(nb,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    (RECORD/'notebook_manifest.json').write_text(json.dumps({'notebook':target.relative_to(ROOT).as_posix(),'sha256':digest(target),'resolution':16,'sigma':0,'epochs':10,'training_core':'unchanged embedded original training.train','data_archive_sha256':delivery['sha256'],'status':'prepared; no GPU training executed'},ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(target,flush=True)

if __name__=='__main__':generate()
