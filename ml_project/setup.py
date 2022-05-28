# https://github.com/pypa/sampleproject/blob/main/setup.py

# Always prefer setuptools over distutils
from setuptools import setup, find_packages
import pathlib

here = pathlib.Path(__file__).parent.resolve()

long_description = (here / "README.md").read_text(encoding="utf-8")


setup(
    name="ml_in_prod_juliarty_ml_project",  # Required
    version="0.1.0",  # Required
    description="That is a homework project",  # Optional
    long_description=long_description,  # Optional
    long_description_content_type="text/markdown",  # Optional (see note above)
    url="https://github.com/made-ml-in-prod-2022/juliarty",  # Optional
    author="Juliarty",  # Optional
    author_email="sirrzk.09@yandex.ru",  # Optional
    keywords="homework ml in prod",  # Optional
    package_dir={"": "src/"},  # Optional
    packages=find_packages(where="src", exclude=["tests"]),  # Required
    package_data={
        # If any package contains *.txt files, include them:
        "": ["configs/*.yaml",
             "configs/features/*.yaml",
             "configs/model/*.yaml",
             "configs/preprocessing/*.yaml",
             "configs/split/*.yaml",
             "configs/train_pipelines/*.yaml"],
    },
    python_requires=">=3.8, <4",
    install_requires=[
        "pyyaml==6.0",
        "marshmallow-dataclass==8.5.8",
        "pandas~=1.4.2",
        "scikit-learn~=1.0.2",
        "numpy~=1.22.3",
        "omegaconf~=2.1.2",
        "hydra-core==1.1.2",
        "hydra_colorlog==1.1.0",
        "pytest~=7.1.2",
        "gdown==4.4.0",
    ],  # Optional
    project_urls={  # Optional
        "Bug Reports": "https://github.com/made-ml-in-prod-2022/juliarty/issues",
        "Source": "https://github.com/made-ml-in-prod-2022/juliarty/",
    },

)
