"""Draw character boxes in reading order and export one line annotation."""
import argparse
import json
import math
from pathlib import Path


def image_box(origin, end, zoom, width, height):
    left, right = sorted((origin[0], end[0]))
    top, bottom = sorted((origin[1], end[1]))
    return [max(0, int(left / zoom)), max(0, int(top / zoom)),
            min(width, math.ceil(right / zoom)), min(height, math.ceil(bottom / zoom))]


def label(image_path, line_id, truth, output, manifest_dir, zoom):
    import tkinter as tk
    from tkinter import messagebox
    from PIL import Image, ImageTk
    if not truth or "".join(truth.split()) != truth:
        raise ValueError("Truth must be nonempty and contain no whitespace")
    image_path = Path(image_path).resolve()
    relative = image_path.relative_to(Path(manifest_dir).resolve()).as_posix()
    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    if zoom <= 0:
        raise ValueError("zoom must be positive")
    root = tk.Tk()
    root.title(f"OCR 字框标注：{line_id}")
    zoomed = ImageTk.PhotoImage(image.resize((round(width * zoom), round(height * zoom))))
    info = tk.StringVar()
    tk.Label(root, textvariable=info, font=("Arial", 14)).pack()
    canvas = tk.Canvas(root, width=round(width * zoom), height=round(height * zoom), cursor="crosshair")
    canvas.pack()
    canvas.create_image(0, 0, anchor="nw", image=zoomed)
    boxes, drawings = [], []
    origin = [0, 0]

    def refresh():
        index = len(boxes)
        info.set(f"{index}/{len(truth)}  下一字：{truth[index] if index < len(truth) else '完成'}    原文：{truth}")

    def start(event):
        origin[:] = [event.x, event.y]

    def finish(event):
        if len(boxes) >= len(truth):
            return
        box = image_box(origin, (event.x, event.y), zoom, width, height)
        if box[0] >= box[2] or box[1] >= box[3]:
            return
        boxes.append(box)
        drawings.append(canvas.create_rectangle(*(v * zoom for v in box), outline="red", width=2))
        refresh()

    def skip():
        if len(boxes) < len(truth):
            boxes.append(None)
            drawings.append(None)
            refresh()

    def undo():
        if boxes:
            boxes.pop()
            drawing = drawings.pop()
            if drawing is not None:
                canvas.delete(drawing)
            refresh()

    def save():
        if len(boxes) != len(truth):
            messagebox.showerror("尚未完成", "请逐字框选，或对无法标框的字符选择跳过。")
            return
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8") as f:
            json.dump({"id": line_id, "image": relative, "truth": truth, "boxes": boxes}, f, ensure_ascii=False, indent=2)
            f.write("\n")
        root.destroy()

    canvas.bind("<ButtonPress-1>", start)
    canvas.bind("<ButtonRelease-1>", finish)
    controls = tk.Frame(root)
    controls.pack()
    for text, command in (("跳过", skip), ("撤销", undo), ("保存", save)):
        tk.Button(controls, text=text, command=command).pack(side="left")
    refresh()
    root.mainloop()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("image")
    p.add_argument("--id", required=True)
    p.add_argument("--truth", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--manifest-dir", required=True)
    p.add_argument("--zoom", type=float, default=2.0)
    a = p.parse_args()
    label(a.image, a.id, a.truth, a.output, a.manifest_dir, a.zoom)


if __name__ == "__main__":
    main()
