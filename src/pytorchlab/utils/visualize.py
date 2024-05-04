from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

__all__ = [
    "make_gif",
]


def make_gif(
    names: list[tuple[str, str]] = [],
    output: str | Path = "output.gif",
    duration: int = 500,
    loop: int = 0,
):
    """
    concrate epoch images into one gif

    Args:
        names (list[tuple[str, str]], optional): name and image pairs. Defaults to [].
        output (str | Path, optional): output path. Defaults to 'output.gif'.
        duration (int, optional): milliseconds for each image. Defaults to 500.
        loop (int, optional): loop times. Defaults to 0.
    """
    names = [(k,Path(v)) for k, v in names]
    image_list: list[Image.Image] = []
    for name, img_path in names:
        img = Image.open(img_path)
        width, height = img.size
        result = Image.new("RGB", (width, height + 20), "white")
        result.paste(img, (0, 20))
        draw = ImageDraw.Draw(result)
        font = ImageFont.load_default(size=16)
        draw.text((0, 0), f"{name}", font=font, fill=(0, 0, 0))
        image_list.append(result)
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    image_list[0].save(
        output,
        format="GIF",
        save_all=True,
        append_images=image_list[1:],
        duration=duration,
        loop=loop,
        optimize=True,
    )
