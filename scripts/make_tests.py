from pathlib import Path
import shutil


def remove_emptydir(path: Path):
    if path.is_file():
        return
    if path.is_dir():
        for child in path.iterdir():
            remove_emptydir(child)
        if not list(path.iterdir()):
            path.rmdir()


def clean_cache(path: Path):
    for x in path.rglob("*"):
        if x.stem == "__pycache__":
            shutil.rmtree(x)


def generate_init_files(path: Path):
    for f in path.rglob("*"):
        if f.is_dir():
            (f / "__init__.py").touch(exist_ok=True)


def make_tests(
    src_root: Path, tests_root: Path, trash_root: Path, pkg_name="pytorchlab"
):
    src_files = [x for x in src_root.rglob("*.py")]
    tests_files = [x for x in tests_root.rglob("*.py")]

    pkg_root = src_root / pkg_name

    for f in src_files:
        aim_f = tests_root / f.relative_to(pkg_root)
        if aim_f.stem != "__init__":
            aim_f = aim_f.with_stem(f"test_{aim_f.stem}")
        aim_f.parent.mkdir(parents=True, exist_ok=True)
        aim_f.touch(exist_ok=True)

    for f in tests_files:
        src_f = pkg_root / f"{f.relative_to(tests_root)}"
        if src_f.name.startswith("test_"):
            src_f = src_f.with_stem(src_f.stem[5:])
        if src_f.exists():
            continue
        trash_to = trash_root / f"{f.relative_to(tests_root)}"
        trash_to.parent.mkdir(parents=True, exist_ok=True)
        f.rename(trash_to)


def main():
    root = Path(__file__).parent.parent
    src_root = root / "src"
    tests_root = root / "tests"
    trash_root = root / "output" / "tests"
    trash_root.mkdir(exist_ok=True, parents=True)

    clean_cache(src_root)
    clean_cache(tests_root)
    generate_init_files(src_root)
    make_tests(src_root, tests_root, trash_root)
    remove_emptydir(tests_root)
    generate_init_files(tests_root)


if __name__ == "__main__":
    main()
