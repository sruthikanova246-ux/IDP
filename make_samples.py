"""Create reproducible demo images in samples/ (all synthetic, generated with Pillow).

  camera_exif.jpg      JPEG with camera EXIF, editing-software tag and a timestamp gap
  no_metadata.png      PNG with no metadata at all
  generator_tags.png   1024x1024 PNG with a Stable-Diffusion-style "parameters" text chunk
  renamed_png.jpg      PNG bytes saved with a .jpg extension
  not_an_image.jpg     Text file with an image extension (should be rejected)

These are test fixtures for the workflow, not real-world evidence: the AI scores the
model gives them say nothing about the model's accuracy on real photographs.
"""
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter
from PIL.PngImagePlugin import PngInfo

OUT = Path(__file__).resolve().parent / "samples"


def scene(size):
    w, h = size
    img = Image.new("RGB", size)
    px = img.load()
    for y in range(h):
        for x in range(w):
            px[x, y] = (int(90 + 120 * y / h), int(140 + 60 * x / w), int(200 - 80 * y / h))
    draw = ImageDraw.Draw(img)
    draw.rectangle([w * 0.1, h * 0.55, w * 0.45, h * 0.95], fill=(120, 80, 60))
    draw.ellipse([w * 0.6, h * 0.1, w * 0.8, h * 0.3], fill=(250, 230, 120))
    draw.polygon([(w * 0.5, h), (w * 0.75, h * 0.45), (w, h)], fill=(70, 110, 70))
    return img.filter(ImageFilter.GaussianBlur(1.2))


def main():
    OUT.mkdir(exist_ok=True)

    exif = Image.Exif()
    exif[0x010F] = "Canon"
    exif[0x0110] = "Canon EOS 5D Mark IV"
    exif[0x0131] = "Adobe Photoshop 25.0 (Windows)"
    exif[0x0132] = "2024:05:02 10:15:00"
    sub = exif.get_ifd(0x8769)
    sub[0x9003] = "2024:05:01 09:30:12"
    sub[0x9004] = "2024:05:01 09:30:12"
    scene((1200, 800)).save(OUT / "camera_exif.jpg", "JPEG", quality=90, exif=exif)

    scene((800, 600)).save(OUT / "no_metadata.png", "PNG")

    info = PngInfo()
    info.add_text("parameters", "a landscape at sunset\nSteps: 30, Sampler: Euler a, CFG scale: 7, "
                                "Size: 1024x1024, Model: sd_xl_base_1.0")
    scene((1024, 1024)).save(OUT / "generator_tags.png", "PNG", pnginfo=info)

    scene((640, 480)).save(OUT / "renamed_png.jpg", "PNG")
    (OUT / "not_an_image.jpg").write_text("This is plain text, not an image.\n", encoding="utf-8")
    for p in sorted(OUT.iterdir()):
        print(f"{p.name:20} {p.stat().st_size:>9,} bytes")


if __name__ == "__main__":
    main()
