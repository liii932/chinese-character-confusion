"""Generate fixed R16 data, package it, then clone the existing Colab notebook."""
import json
from pathlib import Path
import zipfile
from training.prepare_data import build
from training.io_utils import digest, write_json
ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'training/data/lowres16-1140-v1'
RECORD=ROOT/'colab/experiments/lowres16_1140_v1'

def main():
    counts={'train':192,'val_graph':96,'val_select':96,'test':192}
    m=build(DATA,seed=20261030,counts=counts,characters=ROOT/'training/characters1140.json',sigmas=[0],resolution=16)
    expected={s:{'images':n*1140,'per_class':n} for s,n in counts.items()}
    if m['splits']!=expected or m['audit']['unique_pixel_hashes']!=656640:
        raise ValueError('Generated data do not match requested class/split quotas')
    audit={'automatic_checks_passed':True,'classes':1140,'unique_images':m['audit']['unique_pixel_hashes'],
        'splits':m['splits'],'resolution':16,'sigma':0,'cross_split_duplicates':0,'blank_images':0,
        'max_edge_darkness':m['audit']['max_edge_darkness'],
        'method':'Generation-time shared-condition pixel deduplication, clipping, blank, boundary and label construction checks; no additional readability test',
        'readability_basis':'User explicitly confirmed all displayed resolution levels recognizable; blind testing stopped'}
    write_json(DATA/'audit.json',audit)
    readme='''# R16低分辨率同分布基线

1,140类，sigma=0，64清晰图以Lanczos缩至16再bicubic恢复64。训练每类192，两个开发集各96，封存test192，共656,640张。
所有类别使用相同576个无重复位置条件，先做像素去重再分配四集合。训练核心沿用原ResNet18，随机初始化，10epoch、batch64、seed42。
人类可读性以用户人工确认为依据，盲认及评分已停止。新数据、新run均独立保存，旧满分基线保留。
上传本ZIP到MyDrive/hanzi1140_r16，再上传hanzi1140_r16_e10.ipynb到Colab运行三个代码单元。
配套Notebook内嵌训练代码。输出runs/baseline1140_r16_e10_s42_v1，逐轮保存并支持断线恢复；完成后输出val_select与val_graph报告，不自动评价test。
中间分辨率16的重采样可能产生轻微非白边缘；生成时沿用退化协议，要求最外圈暗度<16/255，原字形在高分辨率画布内无裁切。
本数据的划分互斥针对当前随机初始化的R16实验，不能给旧模型提供跨轮独立测试结论。
'''
    (DATA/'README.md').write_text(readme,encoding='utf-8')
    RECORD.mkdir(parents=True,exist_ok=True)
    for name in ['manifest.json','audit.json','README.md']:(RECORD/name).write_bytes((DATA/name).read_bytes())
    destination=ROOT/'colab/dist/hanzi1140_r16_data.zip'
    files=list(m['files'])+['manifest.json','audit.json','README.md']
    with zipfile.ZipFile(destination,'x',zipfile.ZIP_STORED,allowZip64=True) as z:
        for name in files:z.write(DATA/name,'lowres16-1140-v1/'+name)
    delivery={'archive':destination.relative_to(ROOT).as_posix(),'bytes':destination.stat().st_size,'sha256':digest(destination),'files':len(files),'status':'R16 data ready; GPU training not started'}
    write_json(RECORD/'delivery.json',delivery)
    print(json.dumps(delivery,indent=2),flush=True)
    from colab.make_r16_notebook import generate
    generate()

if __name__=='__main__':main()
