from setuptools import setup, find_packages

setup(
    name="slm",
    version="0.1.0",
    # 新フォルダ構成：コードが src/ 以下にある場合の例
    packages=find_packages(where="src"),
    package_dir={'': 'src'},
    install_requires=[
        "torch",
        "numpy",
        "sentencepiece",
        "pywavelets",
        "datasets",
        "huggingface_hub",
        "pytest",
    ],
)