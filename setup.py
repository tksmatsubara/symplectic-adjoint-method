import os
import re
import setuptools

setuptools.setup(
    name="torch_symplectic_adjoint",
    version="0.0.1",
    author="Takashi Matsubara",
    author_email="matsubara@sys.es.osaka-u.ac.jp",
    description="symplectic adjoint method for PyTorch.",
    url="https://github.com/tksmatsubara/symplectic-adjoint-method",
    packages=setuptools.find_packages(),
    install_requires=['torch>=1.3.0'],
    python_requires='~=3.6',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
)
