from setuptools import find_namespace_packages, setup

setup(
    name="big_bertha_v2",
    version="0.2.0",
    packages=find_namespace_packages(include=["big_bertha_v2*"]),
    install_requires=["psutil"],
    python_requires=">=3.10",
)
