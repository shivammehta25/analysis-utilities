import os
from pathlib import Path

import click

from whisperweranalysis.checkpoint_mover import get_list_iter


def remove_checkpoint_in_folder(
    checkpoint_dir, ext, checkpoints_to_keep, filename, splitter
):
    for checkpoint in Path(checkpoint_dir).glob(
        f"{filename.replace('<ast>', '*')}{ext}"
    ):
        if int(checkpoint.stem.split(f"{splitter}")[-1]) not in checkpoints_to_keep:
            click.secho("[-] Deleting: " + str(checkpoint), fg="red")
            checkpoint.unlink()
        else:
            click.secho("[+] Keeping: " + str(checkpoint), fg="green")


@click.command()
@click.option(
    "--dir_path",
    "-i",
    type=click.Path(exists=True),
    required=True,
    help="Path to checkpoint directory where checkpoints will be filtered",
)
@click.option("--ext", "-e", default=".ckpt", help="Extension of checkpoints to filter")
@click.option(
    "--filename",
    "-f",
    default="checkpoint_<ast>",
    help="Filename of checkpoints to filter",
)
@click.option("--splitter", "-s", default="_", help="Split integer on this character")
def main(dir_path, ext, filename, splitter):
    checkpoints_to_keep = get_list_iter()
    for path, _, _ in os.walk(dir_path):
        click.secho("[+] Filtering: " + path, fg="blue")
        remove_checkpoint_in_folder(path, ext, checkpoints_to_keep, filename, splitter)


if __name__ == "__main__":
    main()
