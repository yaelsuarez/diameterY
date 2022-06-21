import os
import pathlib

from setuptools import find_packages, setup


def read(rel_path):
    here = pathlib.Path(__file__).parent.resolve()
    with open(os.path.join(here, rel_path), "r") as fp:
        return fp.read()


def get_version(rel_path):
    for line in read(rel_path).splitlines():
        if line.startswith("__version__"):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    else:
        raise RuntimeError("Unable to find version string.")


setup(
    name="diametery",
    version=get_version("diametery/__init__.py"),
    description="Measuring the diameter of fibers",
    author="(Yael Suarez, Fernando Cossio)",
    author_email="fer_cossio@hotmail.com",
    url="",
    packages=find_packages(),
    package_dir={
        "": ".",
    },
    install_requires=[
        "numpy",
        "tqdm",
    ],
    # package_data={"": ["config.xml", "instruction.html"]},
)
