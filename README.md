<!-- <p align="center"> -->
<!--   <img src="ttt.png" width="60%" alt="TTT-logo"> -->
<!-- </p> -->
<p align="center">
    <h1 align="center">GLYPTIC</h1>
</p>
<p align="center">
    <em>Engraving Fingerprints with Precision and Artistry</em>
</p>
<p align="center">
	<img src="https://img.shields.io/github/license/tariks/glyptic?style=default&logo=opensourceinitiative&logoColor=white&color=0080ff" alt="license">
	<img src="https://img.shields.io/github/last-commit/tariks/glyptic?style=default&logo=git&logoColor=white&color=0080ff" alt="last-commit">
	<img src="https://img.shields.io/github/languages/top/tariks/glyptic?style=default&color=0080ff" alt="repo-top-language">
	<img src="https://img.shields.io/github/languages/count/tariks/glyptic?style=default&color=0080ff" alt="repo-language-count">
<p align="center">
		<em>Built with:</em>
</p>
<p align="center">
	<img src="https://img.shields.io/badge/TensorFlow-FF6F00.svg?style=default&logo=TensorFlow&logoColor=white" alt="TensorFlow">
	<img src="https://img.shields.io/badge/Keras-D00000.svg?style=default&logo=Keras&logoColor=white" alt="Keras">
	<img src="https://img.shields.io/badge/SciPy-8CAAE6.svg?style=default&logo=SciPy&logoColor=white" alt="SciPy">
	<img src="https://img.shields.io/badge/Python-3776AB.svg?style=default&logo=Python&logoColor=white" alt="Python">
	<img src="https://img.shields.io/badge/AIOHTTP-2C5BB4.svg?style=default&logo=AIOHTTP&logoColor=white" alt="AIOHTTP">
</p>

<br>

#####  Table of Contents

- [ Overview](#-overview)
- [ Features](#-features)
- [ Repository Structure](#-repository-structure)
- [ Modules](#-modules)
- [ Getting Started](#-getting-started)
    - [ Installation](#-installation)
    - [ Usage](#-usage)
- [ Contributing](#-contributing)
- [ License](#-license)
- [ Acknowledgments](#-acknowledgments)

---

##  Overview

Written with the jewelry industry in mind, glyptic leverages advanced image processing and generative AI techniques to prepare fingerprint images suitable for jewelry engraving. A two-step process starts by utilizing a pre-trained U-Net model to enhance and denoise raw dactylograms. The second step automates an SDXL-based workflow transforming enhanced images to high-resolution, stylized vector art in SVG format, ensuring compatibility and scalability. Glyptic includes a streamlined setup process that can be deployed on a cloud or local install, and can comfortably run on an M1 Macbook. 

---

##  Features

|    |   Feature         | Description |
|----|-------------------|---------------------------------------------------------------|
| ⚙️  | **Architecture**  | Glyptic follows a modular architecture with a focus on image processing and machine learning, specifically for enhancing fingerprint images using a U-Net model, and image-to-image translation using ComfyUI, SDXL, and controlnet |
| 🔩 | **Code Quality**  | Glyptic follows a structured and organized style with clear separation of concerns, such as modular functions for specific tasks like setup, image enhancement, and image transformation. |
| 🔌 | **Integrations**  | Key integrations include ComfyUI, JuggernautXL, TensorFlow, PyTorch, and Hugging Face Hub, along with image processing libraries like OpenCV, scikit-image, and Pillow. |
| 🧩 | **Modularity**    | The codebase is highly modular, with distinct modules for different functionalities such as image enhancement, setup, and workflow management, promoting reusability. |
| ⚡️  | **Performance**   | Glyptic can both be scaled to run on high-end cloud servers | and run on local machines, with the ability to process large batches of images efficiently. End-to-end processing takes about one minute per image on an M1 macbook with 32GB of RAM. 
| 📦 | **Dependencies**  | The project relies on several key dependencies including TensorFlow, PyTorch, OpenCV, Pillow, and various other libraries for machine learning and image processing. |
| 🚀 | **Simplicity**   | Glyptic wraps up the entire process in a single command, providing a one-click solution out of the box. |

---

##  Repository Structure

```sh
└── glyptic/
    └── src
        └── glyptic
            ├── __init__.py
            ├── enhance_fingerprints.py
            ├── enhance_utils.py
            ├── glyptic_workflow.py
            ├── unet_weights.hdf5
            └── glyptic_setup.py
```

---

##  Modules

<details closed><summary>src.glyptic</summary>

| File | Summary |
| --- | --- |
| [enhance_fingerprints.py](https://github.com/tariks/glyptic/blob/main/src/glyptic/enhance_fingerprints.py) | Enhances fingerprint images by utilizing a pre-trained U-Net model. Downloads necessary model weights if unavailable locally and processes images in batches, resizing and normalizing them before prediction. Outputs enhanced images to a specified directory, optimizing them for further analysis or use within the repository's broader fingerprint processing workflow. |
| [enhance_utils.py](https://github.com/tariks/glyptic/blob/main/src/glyptic/enhance_utils.py) | Defines the U-Net structure used by enhance_fingerprints.py. |
| [unet_weights.hdf5](https://github.com/tariks/glyptic/blob/main/src/glyptic/unet_weights.hdf5) | Pre-trained weights for a U-Net model, facilitating efficient and accurate image processing tasks within the glyptic module. Integrates seamlessly into the workflow, enhancing the systems capability to handle complex image enhancement and fingerprint analysis operations. |
| [glyptic_workflow.py](https://github.com/tariks/glyptic/blob/main/src/glyptic/glyptic_workflow.py) | Facilitates the transformation of input images into high-resolution, stylized outputs using advanced machine learning models and image processing techniques. Integrates various conditioning and control mechanisms to enhance image quality and converts the final output to SVG format, ensuring compatibility and scalability within the repositorys architecture. |
| [glyptic_setup.py](https://github.com/tariks/glyptic/blob/main/src/glyptic_setup/glyptic_setup.py) | Facilitate the setup and configuration of the glyptic application by creating necessary directories, generating configuration files, and optionally downloading required model files. Enhance user experience by providing command-line arguments for custom data directories and automated setup processes. |

</details>

---

##  Getting Started

###  Prerequisites

**Python**: `version 3.11` or higher recommended.
**GPU**: A GPU is recommended for faster processing, but not required.
**RAM**: At least 16GB of RAM is recommended for optimal performance.

Support for cloud deployment is in the works.

###  Installation

```sh
❯ pip install glyptic # or
❯ pipx install glyptic # or
❯ uv tool install glyptic 
```

###  Setup

To configure everything in one command, do
```sh
❯ glyptic_setup --config --download # or
❯ glyptic_setup -c -d 
```

If you already have the required models on hand, skip the download flag and symlink them to your data dir, which is $XDG_DATA_HOME/glyptic by default, or specified by --custom-data-dir.  

Regenerate the config file anytime you update to a new version. 


###  Usage

To run a raw fingerprint scan through the entire workflow, you need only
```sh
❯ glpytic -i myinput/*.jpg 
```

Further options are available: 

```sh
❯ glpytic --help
```

The defaults are good for consistent balance between accuracy and artistry. Support for advanced settings is in the works.

---

##  Contributing

Contributions are welcome! Here are several ways you can contribute:

- **[Report Issues](https://github.com/tariks/glyptic/issues)**: Submit bugs found or log feature requests for the `glyptic` project.
- **[Submit Pull Requests](https://github.com/tariks/glyptic/blob/main/CONTRIBUTING.md)**: Review open PRs, and submit your own PRs.
- **[Join the Discussions](https://github.com/tariks/glyptic/discussions)**: Share your insights, provide feedback, or ask questions.

---

##  License

This project is protected under the [GNU General Public License v3.0](https://choosealicense.com/licenses/gpl-3.0/) License. For more details, refer to the [COPYING](https://github.com/tariks/glyptic/blob/main/COPYING) file.

---

##  Acknowledgments

- [CVxTz/fingerprint_denoising](https://github.com/CVxTz/fingerprint_denoising) for pre-trained model weights
- [Juggernaut-XI](https://huggingface.co/RunDiffusion/Juggernaut-XI-v11) by RunDiffusion
- [ControlNet++](https://huggingface.co/xinsir/controlnet-union-sdxl-1.0)
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI)
- [ComfyScript](https://github.com/Chaoses-Ib/ComfyScript)

---
