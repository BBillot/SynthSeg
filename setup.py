#!/usr/bin/env python3

import re
from setuptools import find_packages, setup

import os
import platform;

py_version = platform.python_version()[:3]

if py_version == "3.6" or py_version == "3.8":

    with open("requirements_python" + py_version + ".txt") as f:
        required_packages = [line.strip() for line in f.readlines()]
        print(required_packages)

    print("Will not build conda module")

    setup(
        name="SynthSeg",
        version="2.0",
        packages=find_packages(),
        author="SynthSeg team",
        description="",
        long_description=open('README.md').read(),
        long_description_content_type='text/markdown',
        license='BSD 3',
        install_requires= required_packages,
        include_package_data=True)
else:
    print("Error, only works with python version 3.6 or 3.8, not {}".format(
        py_version ))

    exit(-1)

