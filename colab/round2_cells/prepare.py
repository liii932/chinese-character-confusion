"""Second-round Colab setup; source text and fixed data hashes are embedded by the generator."""
from pathlib import Path
import hashlib
import json
import shutil
import sys
import zipfile

LOCAL_ROOT = Path('/content')
ARCHIVE_NAME = 'hanzi1140_round2_data.zip'
DATA_MEMBERS = ('manifest.json', 'audit.json', 'labels.json', 'conditions.json',
                'train.npz', 'val_graph.npz', 'val_select.npz', 'test.npz')


def file_sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def prepare_round2(root, local_root, archive_hash, manifest_hash, source_texts):
    root, local_root = Path(root), Path(local_root)
    source_id = hashlib.sha256(json.dumps(source_texts, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
    work = local_root / f'hanzi1140-{archive_hash[:12]}-{source_id[:12]}'
    data = work / 'training/data/round2-1140-v1'
    data.mkdir(parents=True, exist_ok=True)
    archive_path = root / ARCHIVE_NAME
    if not archive_path.is_file():
        raise FileNotFoundError(f'请先将数据包上传到：{archive_path}')
    # Read a local copy during training, retaining the uploaded immutable ZIP on Drive.
    local_zip = local_root / ARCHIVE_NAME
    if not local_zip.is_file() or file_sha256(local_zip) != archive_hash:
        temporary = local_zip.with_suffix('.zip.copying')
        shutil.copyfile(archive_path, temporary)
        if file_sha256(temporary) != archive_hash:
            raise ValueError('数据包 SHA-256 不匹配；请上传本轮原始数据包。')
        temporary.replace(local_zip)
    with zipfile.ZipFile(local_zip) as bundle:
        # Explicit names only: PNG previews and other ZIP entries are not extracted.
        for name in DATA_MEMBERS:
            destination = data / name
            temporary = destination.with_suffix(destination.suffix + '.copying')
            with bundle.open('round2-1140-v1/' + name) as source, temporary.open('wb') as target:
                shutil.copyfileobj(source, target)
            temporary.replace(destination)
    if file_sha256(data / 'manifest.json') != manifest_hash:
        raise ValueError('数据 manifest 与冻结的第二轮版本不一致。')
    manifest = json.loads((data / 'manifest.json').read_text(encoding='utf-8'))
    for name, expected in manifest['files'].items():
        if file_sha256(data / name) != expected:
            raise ValueError(f'数据文件校验失败：{name}')
    counts = {name: value['images'] for name, value in manifest['splits'].items()}
    if counts != {'train': 218880, 'val_graph': 109440, 'val_select': 109440, 'test': 218880}:
        raise ValueError('不是本轮完整 1,140 字数据。')
    label_data = json.loads((data / 'labels.json').read_text(encoding='utf-8'))
    if [r['class_id'] for r in label_data['characters']] != list(range(1140)):
        raise ValueError('类别映射不等于 0～1139。')
    audit = json.loads((data / 'audit.json').read_text(encoding='utf-8'))
    if not audit['automatic_checks_passed'] or audit['unique_images'] != 656640:
        raise ValueError('缺少本轮已完成的像素审计。')
    snapshots = root / 'sources' / source_id
    for name, source in source_texts.items():
        encoded = source.encode('utf-8')
        compile(source, name, 'exec')
        target = work / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(encoded)
        snapshot = snapshots / name
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        if snapshot.exists():
            if snapshot.read_bytes() != encoded:
                raise ValueError(f'Drive 中的源码快照已变化：{snapshot}')
        else:
            snapshot.write_bytes(encoded)
    (root / 'runs').mkdir(exist_ok=True)
    print('数据文件指纹与既有离线审计通过：', counts)
    print('代码和数据位于：', work)
    print('检查点保存到：', root / 'runs')
    return work, data, source_id


if __name__ == '__main__':
    from google.colab import drive
    drive.mount('/content/drive')
    WORK, DATA, SOURCE_ID = prepare_round2(
        DRIVE_ROOT, LOCAL_ROOT, DATA_ARCHIVE_SHA256, DATA_MANIFEST_SHA256, TRAINING_SOURCES)
