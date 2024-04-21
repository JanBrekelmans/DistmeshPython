import os

from setuptools import find_packages, setup

_pkg: str = "DistmeshPython"


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


# Declare minimal set for installation
required_packages = []

setup(
    name=_pkg,
    version="0.1.0",
    description="This package is a port of the Distmesh matlab code into Python. This can be used to generate "
                "unstructured triangular meshes on user defined geometries.",
    author="Jan Brekelmans",
    author_email="j.j.w.c.brekelmans@gmail.com",
    url=f"https://github.com/abcd/{_pkg}/",
    license="MIT",
    keywords="distmesh mesh_generation unstructured_mesh",
    python_requires=">=3.6.0"
)