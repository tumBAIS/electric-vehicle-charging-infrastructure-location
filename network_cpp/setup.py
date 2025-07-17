# Available at setup time due to pyproject.toml
from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

__version__ = "0.0.1"

ext_modules = [
    Pybind11Extension(
        "network_cpp",
        ["src/main.cpp", "src/vertex.cpp", "src/arc.cpp", "src/network.cpp", "src/route.cpp", "src/label.cpp"],
        define_macros=[("VERSION_INFO", __version__)],
    ),
]

setup(
    name="network_cpp",
    version=__version__,
    author="Paul Bischoff",
    author_email="paul.bischoff@tum.de",
    description="A C++ implementation of the labelling algorithm w/ resource constaints",
    long_description="",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    # zip_safe=False,
    python_requires=">=3.7"
)