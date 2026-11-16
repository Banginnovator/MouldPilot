from io import BytesIO
from PIL import Image, ImageDraw
from vision import compare_reference_images


def as_bytes(im):
    b = BytesIO()
    im.save(b, format="PNG")
    return b.getvalue()


def main():
    good = Image.new("RGB", (200, 200), "white")
    d = ImageDraw.Draw(good)
    d.rectangle((50, 40, 150, 160), fill="gray")
    sample = good.copy()
    ImageDraw.Draw(sample).rectangle((110, 40, 150, 160), fill="white")
    result = compare_reference_images(as_bytes(good), as_bytes(sample))
    assert 0 <= result["similarity"] <= 100
    assert result["deviation_pct"] > 0
    assert result["heatmap"].size == (512, 512)
    print("PASS: reference image comparison", result["similarity"], result["deviation_pct"])


if __name__ == "__main__":
    main()
