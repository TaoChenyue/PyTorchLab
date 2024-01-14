from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def find_epoch(s: str):
    index = s.find("epoch")
    if index == -1:
        raise ValueError("no epoch in the string")
    index += 5
    while index < len(s) and not s[index].isdigit():
        index += 1
    if index == len(s):
        raise ValueError("no epoch value in the string")
    end = index
    while end < len(s) and s[end].isdigit():
        end += 1
    result = s[index:end]
    return int(result)


def make_gif(
    save_path: str | Path,
    image_dir: str | Path,
    suffix: str = "png",
    duration: int = 100,
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
    path_list: list[Path] = list(Path(image_dir).glob(f"*.{suffix}"))
    path_list.sort(key=lambda x: find_epoch(str(x)))
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
        draw.text((0, 0), f"{x.stem}", font=font, fill=(0, 0, 0))
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
