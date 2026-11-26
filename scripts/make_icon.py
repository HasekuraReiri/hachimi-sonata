"""生成 Hachimi Sonata 图标（PNG / ICO），供网页 favicon、启动器和 exe 使用。"""
from __future__ import annotations

import os

from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FONT = r"C:\Windows\Fonts\msyhbd.ttc"
SIZE = 1024


def diagonal_gradient(size: int) -> Image.Image:
    img = Image.new("RGBA", (size, size))
    px = img.load()
    c1 = (124, 92, 255)
    c2 = (255, 96, 178)
    for y in range(size):
        for x in range(size):
            t = (x / size + y / size) / 2
            px[x, y] = (
                int(c1[0] + (c2[0] - c1[0]) * t),
                int(c1[1] + (c2[1] - c1[1]) * t),
                int(c1[2] + (c2[2] - c1[2]) * t),
                255,
            )
    return img


def main() -> None:
    canvas = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
    grad = diagonal_gradient(SIZE)
    mask = Image.new("L", (SIZE, SIZE), 0)
    ImageDraw.Draw(mask).rounded_rectangle((18, 18, SIZE - 18, SIZE - 18), radius=210, fill=255)
    canvas.paste(grad, (0, 0), mask)

    draw = ImageDraw.Draw(canvas)
    # 柔和内描边
    draw.rounded_rectangle((30, 30, SIZE - 30, SIZE - 30), radius=198, outline=(255, 255, 255, 46), width=8)

    font = ImageFont.truetype(FONT, 600)
    text = "哈"
    box = draw.textbbox((0, 0), text, font=font)
    tw, th = box[2] - box[0], box[3] - box[1]
    x = (SIZE - tw) / 2 - box[0]
    y = (SIZE - th) / 2 - box[1] - 24
    # 轻微投影，增强小尺寸辨识度
    draw.text((x + 10, y + 14), text, font=font, fill=(60, 20, 100, 90))
    draw.text((x, y), text, font=font, fill=(255, 255, 255, 255))

    # 底部三条音波，让图标更像“音律工坊”
    for i, (ww, hh) in enumerate(((70, 34), (110, 58), (54, 26))):
        left = SIZE // 2 - 170 + i * 130
        top = SIZE - 170 - hh
        draw.rounded_rectangle((left, top, left + ww, top + hh), radius=hh // 2, fill=(255, 255, 255, 235))

    canvas.save(os.path.join(ROOT, "HachimiSonata.png"))
    canvas.save(
        os.path.join(ROOT, "HachimiSonata.ico"),
        sizes=[(256, 256), (128, 128), (64, 64), (48, 48), (32, 32), (16, 16)],
    )
    print("icon written:", os.path.join(ROOT, "HachimiSonata.ico"))


if __name__ == "__main__":
    main()
