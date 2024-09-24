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


def write_config(data_path, f):
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
    (data_path / "models" / "clip").mkdir(parents=True, exist_ok=True)
    (data_path / "models" / "clip_vision").mkdir(parents=True, exist_ok=True)
    (data_path / "models" / "configs").mkdir(parents=True, exist_ok=True)
    (data_path / "models" / "controlnet").mkdir(parents=True, exist_ok=True)
    (data_path / "models" / "diffusion_models").mkdir(parents=True, exist_ok=True)
    (data_path / "models" / "unet").mkdir(parents=True, exist_ok=True)
    (data_path / "models" / "embeddings").mkdir(parents=True, exist_ok=True)
    (data_path / "models" / "loras").mkdir(parents=True, exist_ok=True)
    (data_path / "models" / "upscale_models").mkdir(parents=True, exist_ok=True)
    (data_path / "models" / "vae").mkdir(parents=True, exist_ok=True)
    print(f"model dirs prepared at {data_path.resolve()}")
    if download:
        import urllib.request

        print("downloading Juggernaut v11")
        print("downloading controlnet-union")
