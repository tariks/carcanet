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
    sparams: tuple[float, float],
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
            model = SelfAttentionGuidance(model, *sparams)
        conditioning = CLIPTextEncodeSDXL(
            832,
            1152,
            0,
            0,
            832,
            1152,
            f"(fingerprint:1.2), (medium-detail line art suitable for CNC machining), (black lines on white background), (consistent depth engraving), smooth and continuous lines, (casting-ready design), (lost-wax casting style), machinable design, engraving-ready vector, high-resolution{pos}",
            clip,
            f"(fingerprint:1.2), (medium-detail line art suitable for CNC machining), (black lines on white background), (consistent depth engraving), smooth and continuous lines, (casting-ready design), (lost-wax casting style), machinable design, engraving-ready vector, high-resolution{pos}",
        )
        conditioning2 = CLIPTextEncodeSDXL(
            832,
            1152,
            0,
            0,
            832,
            1152,
            f"faded, gradient, dotted, dots, grey fill, color, thin elements, undercuts, overhangs, fragile structures, excessive depth variation, delicate patterns, non-manufacturable, disconnected lines, rough lines, noise, distortion, pixelation",
            clip,
            f"faded, gradient, dotted, dots, grey fill, color, thin elements, undercuts, overhangs, fragile structures, excessive depth variation, delicate patterns, non-manufacturable, disconnected lines, rough lines, noise, distortion, pixelation",
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
