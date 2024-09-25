import sys
import argparse
from contextlib import contextmanager
import shutil
import os
from pathlib import Path

internal_path = Path(__file__).parent / "internal"
internal_input = internal_path / "input"
internal_input.mkdir(parents=True, exist_ok=True)


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


def refine(infiles: list[Path], outdir: Path):
    from .enhance_fingerprints import enhance

    enhance(infiles, outdir)


def run_diffusion(infile: Path, outdir: Path, args: argparse.Namespace):
    from .glyptic_workflow import img2img

    img2img(
        infile,
        outdir,
        no_sag=args.no_sag,
        sagparams=args.sagparams,
        pos=args.pos,
        neg=args.neg,
        control_type=args.control_type,
        control_strength=args.control_strength,
        steps=args.steps,
        cfg=args.cfg,
        sampler=args.sampler,
    )


def glyptic():

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
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
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
    adv_group = parser.add_argument_group(
        title="Advanced options",
        description="provides more control over the diffusion process",
    )
    adv_group.add_argument(
        "--steps", type=int, default=8, help="Number of diffusion steps"
    )
    adv_group.add_argument(
        "--cfg", type=float, default=4, help="Classifier free guidance scale"
    )
    adv_group.add_argument(
        "--sampler",
        type=str,
        default="euler_ancestral",
        help="Sampler type",
        choices=["euler_ancestral", "dpmpp_2m", "dpmpp_2m_sde"],
    )
    adv_group.add_argument(
        "--no-sag",
        action="store_true",
        help="Disable self-attention guidance. Useful for speeding things up.",
    )
    adv_group.add_argument(
        "--sagparams",
        type=float,
        nargs=2,
        default=(0.5, 2),
        help="Self-attention guidance. Expects two values, ex: --sagparams <scale> <blur_sigma>",
    )
    adv_group.add_argument(
        "--control-strength",
        type=float,
        default=0.5,
        help="Controlnet strength. Reducing may compensate for low quality input. Increasing may enhance higher quality input. ",
    )
    adv_group.add_argument(
        "--control-type",
        type=str,
        default="depth",
        choices=["depth", "auto", "repaint"],
        help="Controlnet mode.",
    )
    adv_group.add_argument(
        "--pos", type=str, default="", help="Append additional text to positive prompt"
    )
    adv_group.add_argument(
        "--neg", type=str, default="", help="Append additional text to negative prompt"
    )

    args = parser.parse_args()

    infiles = [Path(f).resolve() for f in args.input]
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    if args.enhance_only:
        refine(infiles, outdir)
    elif args.diffusion_only:
        # with suppress_stdout():
        for infile in infiles:
            shutil.copy2(infile, internal_input)
            if args.verbose:
                print(f"Running diffusion on {infile.name}")
                run_diffusion(infile, outdir, args)
            else:
                with suppress_stdout():
                    run_diffusion(infile)
    else:
        refine(infiles, internal_input)

        for infile in internal_input.glob("*png"):
            if args.verbose:
                print(f"Running diffusion on {internal_input}")
                run_diffusion(infile, outdir, args)
            else:
                with suppress_stdout():
                    run_diffusion(infile)


def glyptic_setup():
    from .glyptic_setup import glyptic_setup as gs
    from xdg_base_dirs import xdg_data_home

    gdata = xdg_data_home() / "glyptic"
    parser = argparse.ArgumentParser(
        prog="glyptic_setup",
        description="Set up or download required data for glyptic.",
        epilog="See https://github.com/tariks/glyptic for more info.",
    )
    parser.add_argument(
        "-d",
        "--download",
        action="store_true",
        help="Download required models (about 9GB). If you already have them, better to skip this and symlink them to the data dirs",
    )
    parser.add_argument(
        "-c",
        "--config",
        action="store_true",
        help="Create config file user XDG_CONFIG/glyptic directory. This file defines where glyptic looks for models. If it already exists, it will be overwritten.",
    )
    parser.add_argument(
        "--custom-data-dir",
        type=str,
        default=str(gdata),
        help="Specify the base data directory. If not specified, will use XDG_DATA_HOME/glyptic. Use in conjunction with --config and/or --download.",
    )

    args = parser.parse_args()

    gs(args)
