from setuptools import find_packages, setup


setup(
    name="diametery",
    version='0.1.0',
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