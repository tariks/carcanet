import sys
from contextlib import contextmanager
import shutil
import os
from pathlib import Path

internal_path = Path(__file__).parent / "internal"
internal_input = internal_path / "input"


@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = devnull
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr


import argparse

parser = argparse.ArgumentParser(
    prog="glyptic",
    description="A tool to turn fingerprint scans into vector art.",
    epilog="For more information, visit https://github.com/tariks/glyptic",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(
    "-i",
    "--input",
    nargs="+",
    required=True,
    action="extend",
    help="Path to input fingerprint image(s). Can be a space-delimited sequence of file paths or a glob expression (myinput/*.jpg, etc)",
)
parser.add_argument("-o", "--outdir", default="output", help="Output directory")
group = parser.add_mutually_exclusive_group()
group.add_argument(
    "--enhance-only",
    action="store_true",
    help="Only perform enhancement preprocessing, skip stable diffusion. Expects raw fingerprint jpegs. Outputs 288x400 greyscale pngs.",
)
group.add_argument(
    "--diffusion-only",
    action="store_true",
    help="Only perform diffusion step, skip enhancement preprocessing. Expects enhanced 288x400 pngs. Outputs high resolution line art as png and svg files.",
)
args = parser.parse_args()

infiles = [Path(f).resolve() for f in args.input]
outdir = Path(args.outdir).resolve()
outdir.mkdir(parents=True, exist_ok=True)


def refine(infiles=infiles, outdir=outdir):
    from .enhance_fingerprints import enhance

    enhance(infiles, outdir)


def glyptic():
    if args.enhance_only:
        refine(infiles, outdir)
    elif args.diffusion_only:
        with suppress_stdout():
            from .glyptic_workflow import run_workflow

            for infile in infiles:
                shutil.copy2(infile, internal_input)
                run_workflow(infile, outdir)
    else:
        refine(infiles, internal_input)
        with suppress_stdout():
            from .glyptic_workflow import run_workflow

            for infile in internal_input.glob("*png"):
                run_workflow(infile, outdir)
