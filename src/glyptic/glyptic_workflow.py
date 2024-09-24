import sys
from pathlib import Path

internal = Path(__file__).parent / "internal"
sys.path.insert(0, str(internal))
from comfy_script.runtime import *

load()
from comfy_script.runtime.nodes import *


import vtracer


def run_workflow(infile, outdir):
    basefull = infile.name
    base = infile.stem
    outfile = outdir / f"{base}.png"
    with Workflow():
        model, clip, vae = CheckpointLoaderSimple(
            "Juggernaut-XI-byRunDiffusion.safetensors"
        )
        model = SelfAttentionGuidance(model, 0.5, 2)
        conditioning = CLIPTextEncodeSDXL(
            832,
            1152,
            0,
            0,
            832,
            1152,
            "fingerprint, line art, black lines white background, (high-resolution image:1.3).",
            clip,
            "fingerprint, line art, black lines white background, (high-resolution image:1.3).",
        )
        conditioning2 = CLIPTextEncodeSDXL(
            832,
            1152,
            0,
            0,
            832,
            1152,
            "faded, blurry, grey fill, dotted.",
            clip,
            "faded, blurry, grey fill, dotted.",
        )
        control_net = ControlNetLoader(
            "xinsir_controlnet-union-sdxl-1.0/diffusion_pytorch_model_promax.safetensors"
        )
        control_net = SetUnionControlNetType(control_net, "depth")
        image, _ = LoadImage(basefull)
        image = ImageScale(image, "bicubic", 832, 1152, "disabled")
        image2 = ImageInvert(image)
        positive, negative = ControlNetApplyAdvanced(
            conditioning, conditioning2, control_net, image2, 0.5, 0, 1
        )
        latent = VAEEncode(image, vae)
        latent = KSamplerAdvanced(
            model,
            "enable",
            42,
            8,
            4,
            "euler_ancestral",
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
