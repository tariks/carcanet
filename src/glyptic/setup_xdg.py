from xdg_base_dirs import xdg_config_home, xdg_data_home
from pathlib import Path

gdata = xdg_data_home / "glyptic"


def make_config(data_path=gdata):
    (xdg_config_home / "glyptic").mkdir(parents=True, exist_ok=True)
    yaml = xdg_config_home / "glyptic" / "extra_model_paths.yaml"
    with yaml.open("w") as f:
        write_config(data_path, f)
    yaml.symlink_to(Path(__file__) / "internal" / "extra_model_paths.yaml")
    make_data_dirs(data_path, download=False)


def write_config(data_path=gdata, f):
    print("comfyui:", file=f)
    print(f"  base_path: {str(data_path)}", file=f)
    print("  checkpoints: models/checkpoints/", file=f)
    print("  clip: models/clip/", file=f)
    print("  clip_vision: models/clip_vision/", file=f)
    print("  configs: models/configs/", file=f)
    print("  controlnet: models/controlnet/", file=f)
    print("  diffusion_models = |", file=f)
    print("    models/diffusion_models", file=f)
    print("    models/unet", file=f)
    print("  embeddings: models/embeddings/", file=f)
    print("  loras: models/loras/", file=f)
    print("  upscale_models: models/upscale_models/", file=f)
    print("  vae: models/vae/", file=f)


def make_data_dirs(data_path=gdata, download=True):
    (data_path / "models" / "checkpoints").mkdir(parents=True, exist_ok=True)
    (data_path / "models" / "clip").mkdir( exist_ok=True)
    (data_path / "models" / "clip_vision").mkdir( exist_ok=True)
    (data_path / "models" / "configs").mkdir( exist_ok=True)
    (data_path / "models" / "controlnet").mkdir( exist_ok=True)
    (data_path / "models" / "diffusion_models").mkdir( exist_ok=True)
    (data_path / "models" / "unet").mkdir( exist_ok=True)
    (data_path / "models" / "embeddings").mkdir( exist_ok=True)
    (data_path / "models" / "loras").mkdir( exist_ok=True)
    (data_path / "models" / "upscale_models").mkdir( exist_ok=True)
    (data_path / "models" / "vae").mkdir( exist_ok=True)
    print(f"model dirs prepared at {data_path.resolve()}")


def download_models(data_path=gdata):
    from huggingface_hub import hf_hub_download
    print("downloading models...")
    print("downloading Juggernaut-XI-v11")
    hf_hub_download(
        repo_id="RunDiffusion/Juggernaut-XI-v11",
        filename="Juggernaut-XI-byRunDiffusion.safetensors",
        local_dir=data_path / "models" / "checkpoints"
    )
    controlnet = data_path / "models" / "controlnet" / "xinsir_controlnet-union-sdxl-1.0"
    controlnet.mkdir(exist_ok=True, parents=True)
    hf_hub_download(
        repo_id="xinsir/controlnet-union-sdxl-1.0",
        filename="diffusion_pytorch_model_promax.safetensors",
        local_dir=controlnet
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
    action="store_false",
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
    default=gdata.as_posix(),
    help="Specify the base data directory. If not specified, will use XDG_DATA_HOME/glyptic. Use in conjunction with --config and/or --download."
)

args = parser.parse_args()

def glyptic_setup():
    if args.config:
        make_config(Path(args.custom_data_dir))
        print(f"config file created at {str(xdg_config_home / 'glyptic' / 'extra_model_paths.yaml')}")
    if args.download:
        download_models(Path(args.custom_data_dir))
        print(f"completed downloads to {str(xdg_data_home / 'glyptic' / 'models')}")

if __name__ == "__main__":
    glyptic_setup()
