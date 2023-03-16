import shutil
from pathlib import Path

from tqdm.auto import tqdm

CKPT_EXTENTION = ".pth"


def get_glow_iteration(path):
    return int(path.stem.split("_")[-1])


def copy_checkpoints(checkpoint_dir, subfolder_dir, list_iters):
    """Copy all checkpoints from checkpoint_dir to subfolder_dir."""
    if not isinstance(checkpoint_dir, Path):
        checkpoint_dir = Path(checkpoint_dir)
    if not isinstance(subfolder_dir, Path):
        subfolder_dir = Path(subfolder_dir)

    Path(checkpoint_dir / subfolder_dir).mkdir(parents=True, exist_ok=True)
    subfolder_dir = checkpoint_dir / subfolder_dir
    all_checkpoints = checkpoint_dir.glob(f"*{CKPT_EXTENTION}")
    for filename in tqdm(all_checkpoints):
        iteration = get_glow_iteration(
            filename
        )  # TODO: change here for different systems
        if iteration in list_iters:
            shutil.copy(filename, subfolder_dir / filename.name)


def get_list_iter():
    list_ = []
    for i in range(0, 500500, 500):
        if i <= 2500:
            list_.append(i)
        elif i <= 30000 and i % 5000 == 0:
            list_.append(i)
        elif i <= 100000 and i % 10000 == 0:
            list_.append(i)
        elif i <= 500000 and i % 50000 == 0:
            list_.append(i)

    return set(list_)


if __name__ == "__main__":
    list_iters = get_list_iter()
    print(len(list_iters))
    print(list_iters)
    copy_checkpoints("./logs/blank", "glow_checkpoints", list_iters)
