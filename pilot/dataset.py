import json
import hashlib
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset
import numpy as np
ROOT = Path(__file__).resolve().parent

class CharDataset(Dataset):
    def __init__(self, split: str, root_dir: Path = ROOT / "dataset_81_v2"):
        """
        :param split: 选择集合："train", "val_graph", "val_select", "test"
        :param root_dir: 数据集根目录
        """
        if split not in {"train", "val_graph", "val_select", "test"}:
            raise ValueError(f"Unknown split: {split}")
        self.root_dir = Path(root_dir)
        manifest_path = self.root_dir / "meta/manifest.json"
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if self.manifest.get("schema_version") != 2:
            raise ValueError("Regenerate data with the repaired build_dataset.py")
        raw = (self.root_dir / "meta/dataset_meta.json").read_bytes()
        if hashlib.sha256(raw).hexdigest() != self.manifest["metadata_sha256"]:
            raise ValueError("Dataset metadata hash mismatch")
        all_records = json.loads(raw)
        hashes = [r["pixel_sha256"] for r in all_records]
        if len(hashes) != len(set(hashes)):
            raise ValueError("Duplicate pixels across dataset records")

        # 筛选属于当前split的样本
        self.records = [r for r in all_records if r["split"] == split]
        if not self.records:
            raise ValueError(f"Empty dataset split: {split}")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        img_path = self.root_dir / rec["file_relative"]

        # 读取灰度图
        with Image.open(img_path) as source:
            if source.mode != "L" or source.size != (64, 64):
                raise ValueError(f"Expected 64x64 grayscale: {img_path}")
            img = source.copy()
        if hashlib.sha256(img.tobytes()).hexdigest() != rec["pixel_sha256"]:
            raise ValueError(f"Image pixels differ from metadata: {img_path}")
        # 转为numpy，再tensor；shape [1,64,64]
        img_tensor = torch.from_numpy(np.array(img, dtype=np.float32))
        img_tensor = img_tensor.unsqueeze(0)

        # 归一化，和设计书v0.6约定：(x/255 -0.5)/0.5  → [-1 ~ +1]
        img_tensor = (img_tensor / 255.0 - 0.5) / 0.5

        label = int(rec["class_id"])
        return img_tensor, label


if __name__ == "__main__":
    import numpy as np
    # 简单自测，读取少量样本看是否正常
    ds = CharDataset(split="train")
    print(f"train样本数量：{len(ds)}")
    img, lab = ds[0]
    print(f"图片shape: {img.shape}")
    print(f"标签class_id: {lab}")
    print(f"像素值域 min={img.min().item():.3f}, max={img.max().item():.3f}")
