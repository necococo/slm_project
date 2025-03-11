from setuptools import setup, find_packages

setup(
    name="slm",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        # "torch",
        # "transformers",
        "pywavelets",
        "datasets",
        # "huggingface_hub",
        "bitsandbytes>=0.41.0,<0.42.0",
        # "fugashi",
        "ipadic",
        "unidic-lite",
        "cut-cross-entropy",
        "optuna"
    ],
    extras_require={
        "dev": [
            "pytest",
        ],
    },
)