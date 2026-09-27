#!/usr/bin/env python

import setuptools

with open('requirements.txt') as f:
    required_packages = [line.strip() for line in f.readlines() if line.strip()]

setuptools.setup(name='SynthSeg',
                 version='2.0',
                 license='Apache 2.0',
                 description='Domain-agnostic segmentation of brain scans',
                 author='Benjamin Billot',
                 url='https://github.com/BBillot/SynthSeg',
                 keywords=['segmentation', 'domain-agnostic', 'brain'],
                 packages=setuptools.find_packages(),
                 python_requires='>=3.9,<3.12',
                 install_requires=required_packages,
                 include_package_data=True)
