from xdg_base_dirs import xdg_config_home, xdg_data_home
from pathlib import Path
from typing import TextIO

gdata = xdg_data_home() / "glyptic"


def make_config(data_path: Path):
    glyptic = xdg_config_home() / "glyptic"
    glyptic.mkdir(parents=True, exist_ok=True)
    yaml = glyptic / "extra_model_paths.yaml"
    symyaml = Path(__file__).parent / ".." / "glyptic" / "internal" / "extra_model_paths.yaml"
    yaml.unlink(missing_ok=True)
    with yaml.open("w") as f:
        write_config(data_path, f)
    symyaml.unlink(missing_ok=True)
    symyaml.symlink_to(yaml.resolve())
    make_data_dirs(data_path)


def write_config(data_path: Path, f: TextIO):
    print(f'''comfyui:
    base_path: {str(data_path.resolve())}
    is_default: true
    checkpoints: models/checkpoints/
    clip: models/clip/
    clip_vision: models/clip_vision/
    configs: models/configs/
    controlnet: models/controlnet/
    diffusion_models: |
                 models/diffusion_models
                 models/unet
    embeddings: models/embeddings/
    loras: models/loras/
    upscale_models: models/upscale_models/
    vae: models/vae/

''', file = f)


def make_data_dirs(data_path: Path):
    (data_path / "models" / "checkpoints").mkdir(parents=True, exist_ok=True)
    (data_path / "models" / "clip").mkdir(exist_ok=True)
    (data_path / "models" / "clip_vision").mkdir(exist_ok=True)
    (data_path / "models" / "configs").mkdir(exist_ok=True)
    (data_path / "models" / "controlnet").mkdir(exist_ok=True)
    (data_path / "models" / "diffusion_models").mkdir(exist_ok=True)
    (data_path / "models" / "unet").mkdir(exist_ok=True)
    (data_path / "models" / "embeddings").mkdir(exist_ok=True)
    (data_path / "models" / "loras").mkdir(exist_ok=True)
    (data_path / "models" / "upscale_models").mkdir(exist_ok=True)
    (data_path / "models" / "vae").mkdir(exist_ok=True)
    print(f"model dirs prepared at {data_path.resolve()}")


def download_models(data_path: Path):
    from huggingface_hub import hf_hub_download

    print("downloading models...")
    # print("downloading Juggernaut-XI-v11")
    # hf_hub_download(
    #     repo_id="RunDiffusion/Juggernaut-XI-v11",
    #     filename="Juggernaut-XI-byRunDiffusion.safetensors",
    #     local_dir=data_path / "models" / "checkpoints",
    # )
    controlnet = (
        data_path / "models" / "controlnet" / "xinsir_controlnet-union-sdxl-1.0"
    )
    print("downloading controlnet-union...")
    controlnet.mkdir(exist_ok=True, parents=True)
    hf_hub_download(
        repo_id="xinsir/controlnet-union-sdxl-1.0",
        filename="diffusion_pytorch_model_promax.safetensors",
        local_dir=controlnet,
    )


import argparse

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


def glyptic_setup():
    data_path = Path(args.custom_data_dir).resolve()
    if args.config:
        make_config(data_path)
        print(
            f"config file created at {str(xdg_config_home() / 'glyptic' / 'extra_model_paths.yaml')}"
        )
    if args.download:
        download_models(data_path)
        print(f"completed downloads to {str(xdg_data_home() / 'glyptic' / 'models')}")
