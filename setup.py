from setuptools import setup, find_packages

setup(
    name="slm",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch",
        "numpy",
        "sentencepiece",
        "pywavelets",
        "datasets",
        "huggingface_hub",
        "pytest",
        "bitsandbytes",
        "fugashi",
        "ipadic",
        "unidic-lite",
        "cut-cross-entropy",
    ],
)