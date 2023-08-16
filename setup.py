from setuptools import setup

setup(
    name="distill-diffusion",
    py_modules=["model", "evaluations, dataset"],
    install_requires=[
        "torch",
        "tqdm",
        "numpy",
        "matplotlib",
        "huggingface_hub",
        "diffusers",
        "accelerate",
        "transformers",
        "pytorch-lightning",
        "tensorboard",
        "webdataset",
        "clip"
        ""
    ],
)
