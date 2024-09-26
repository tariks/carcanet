import sys
from pathlib import Path
import argparse

internal = Path(__file__).parent / "internal"
sys.path.insert(0, str(internal))
from comfy_script.runtime import *

load()
from comfy_script.runtime.nodes import *


import vtracer


def img2img(
    infile: Path,
    outdir: Path,
    no_sag: bool,
    sagparams: tuple[float, float],
    pos: str,
    neg: str,
    control_type: str,
    control_strength: float,
    steps: int,
    cfg: float,
    sampler: str,
):
    basefull = infile.name
    base = infile.stem
    outfile = outdir / f"{base}.png"
    with Workflow():
        model, clip, vae = CheckpointLoaderSimple(
            "Juggernaut-XI-byRunDiffusion.safetensors"
        )
        if not no_sag:
            model = SelfAttentionGuidance(model, *sagparams)
        conditioning = CLIPTextEncodeSDXL(
            832,
            1152,
            0,
            0,
            832,
            1152,
            f"High-Resolution detailed line art of ((fingerprint)), black lines on white background, (precise and elegant lines), (elements of arabic calligraphy style), (crisp digital vector art), (optimized for laser engraving), sharp details{pos}",
            clip,
            f"High-Resolution detailed line art of ((fingerprint)), black lines on white background, (precise and elegant lines), (elements of arabic calligraphy style), (crisp digital vector art), (optimized for laser engraving), sharp details{pos}",
        )
        conditioning2 = CLIPTextEncodeSDXL(
            832,
            1152,
            0,
            0,
            832,
            1152,
            f"faded, blurry, grey fill, gradient, dotted, low-resolution, color, shading, text{neg}",
            clip,
            f"faded, blurry, grey fill, gradient, dotted, low-resolution, color, shading, text{neg}",
        )
        control_net = ControlNetLoader(
            "xinsir_controlnet-union-sdxl-1.0/diffusion_pytorch_model_promax.safetensors"
        )
        control_net = SetUnionControlNetType(control_net, control_type)
        image, _ = LoadImage(basefull)
        image = ImageScale(image, "bicubic", 832, 1152, "disabled")
        image2 = image if control_type == "repaint" else ImageInvert(image)
        positive, negative = ControlNetApplyAdvanced(
            conditioning, conditioning2, control_net, image2, control_strength, 0, 1
        )
        latent = VAEEncode(image, vae)
        latent = KSamplerAdvanced(
            model,
            "enable",
            42,
            steps,
            cfg,
            sampler,
            "karras",
            positive,
            negative,
            latent,
            0,
            10000,
            "enable",
        )
        image3 = VAEDecode(latent, vae)
        image = SaveImage(image3, base)
        image = image.wait()[0]
        image.save(outfile, optimize=True)
        vtracer.convert_image_to_svg_py(
            outfile.as_posix(),
            outfile.with_suffix(".svg").as_posix(),
            colormode="binary",
        )
        for i in (internal / "output").glob(f"{base}*g"):
            i.unlink()
        (internal / "input" / basefull).unlink(missing_ok=True)
