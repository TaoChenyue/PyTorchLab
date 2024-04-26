from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def make_gif(
    root: str|Path,
    name: str='output.png',
    save_path: str | Path='output.gif',
    duration: int = 500,
    loop: int = 0,
    start: int = 0,
    end: int | None = None,
    step: int = 1,
):
    """
    concrate epoch images into one gif

    Args:
        save_path (str | Path): path to save gif
        image_dir (str | Path): directory path of images
        suffix (str, optional): image suffix. Defaults to "png".
        duration (int, optional): duration to display each image. Defaults to 100.
        loop (int, optional): replay times when finishing. Defaults to 0.
        start (int, optional): start index of image list. Defaults to 0.
        end (int | None, optional): end index of image list. Defaults to None.
        step (int, optional): skip images with step. Defaults to 1.
    """
    root = Path(root)
    epoch_list: list[Path] = sorted([x for x in root.iterdir()],key=lambda x: int(x.stem.split('=')[-1]))
    path_list: list[Path] = [x/name for x in epoch_list]
    if end is None:
        end = len(path_list)
    path_list = path_list[start:end:step] + [path_list[end - 1]]
    image_list: list[Image.Image] = []
    for x in path_list:
        img = Image.open(x)
        width, height = img.size
        result = Image.new("RGB", (width, height + 20), "white")
        result.paste(img, (0, 20))
        draw = ImageDraw.Draw(result)
        font = ImageFont.load_default(size=16)
        draw.text((0, 0), f"{x.parent.stem}", font=font, fill=(0, 0, 0))
        image_list.append(result)
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    image_list[0].save(
        save_path,
        format="GIF",
        save_all=True,
        append_images=image_list[1:],
        duration=duration,
        loop=loop,
        optimize=True,
    )


if __name__ == "__main__":
    from jsonargparse import CLI

    CLI(make_gif)