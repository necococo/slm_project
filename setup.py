from setuptools import setup, find_packages

setup(
    name="slm",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "sentencepiece",
        "pywavelets",
        "datasets",
        "huggingface_hub",
        "pytest",
        "bitsandbytes>=0.41.0,<0.42.0",
        "fugashi",
        "ipadic",
        "unidic-lite",
        "cut-cross-entropy",
    ],
)