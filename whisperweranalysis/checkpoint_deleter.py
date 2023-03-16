from pathlib import Path

import click

from whisperweranalysis.checkpoint_mover import get_list_iter


@click.command()
@click.option(
    "--checkpoint_dir",
    "-i",
    type=click.Path(exists=True),
    required=True,
    help="Path to checkpoint directory where checkpoints will be filtered",
)
@click.option("--ext", "-e", default=".ckpt", help="Extension of checkpoints to filter")
def main(checkpoint_dir, ext):
    checkpoints_to_keep = get_list_iter()
    for checkpoint in Path(checkpoint_dir).glob(f"*{ext}"):
        if int(checkpoint.stem.split("_")[-1]) not in checkpoints_to_keep:
            click.secho("[-] Deleting: " + str(checkpoint), fg="red")
            checkpoint.unlink()
        else:
            click.secho("[+] Keeping: " + str(checkpoint), fg="green")


if __name__ == "__main__":
    main()
