"""Package only allowlisted training files, the approved font and checked data."""
import json
from pathlib import Path
import zipfile

from training.io_utils import digest, verify_dataset

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "training/data/baseline81-v2"
verify_dataset(DATA)
names = [
    "pilot/characters.json", "pilot/render_samples.py", "pilot/requirements.txt",
    "pilot/fonts/SourceHanSansSC-Regular.otf", "pilot/fonts/LICENSE.txt", "pilot/fonts/source.json",
    "training/__init__.py", "training/prepare_data.py", "training/io_utils.py",
    "training/train.py", "training/requirements-colab.txt", "training/test_pipeline.py",
    "colab/README.md", "colab/hanzi81_baseline.ipynb", "colab/VALIDATION.md",
    "colab/make_notebook.py", "colab/build_bundle.py",
]
names += [f"training/data/baseline81-v2/{name}" for name in (
    "manifest.json", "conditions.json", "labels.json", "train.npz", "val_graph.npz", "val_select.npz", "test.npz")]
manifest = {"schema": 1, "files": {name: digest(ROOT / name) for name in names}}
destination = ROOT / "colab/dist/hanzi81_colab_bundle.zip"
destination.parent.mkdir(exist_ok=True)
temporary = destination.with_suffix(".zip.tmp")
with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
    for name in names:
        archive.write(ROOT / name, name)
    archive.writestr("bundle_manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2))
temporary.replace(destination)
print(json.dumps({"file": str(destination), "MiB": round(destination.stat().st_size / 2**20, 2),
                  "sha256": digest(destination), "files": len(names)}, ensure_ascii=False, indent=2))
