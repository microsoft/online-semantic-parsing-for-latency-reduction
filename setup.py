#!/usr/bin/env python
#  Copyright (c) Microsoft Corporation.
#  Licensed under the MIT license.
from setuptools import find_packages, setup

setup(
    name="online-semantic-parsing",
    version="0.1",
    author="Jiawei Zhou and Sam Thomson",
    author_email="jzhou02@g.harvard.edu",
    description="Online Semantic Parsing for Latency Reduction in Task-Oriented Dialogue",
    license="MIT",
    packages=find_packages("src"),
    package_dir={"": "src"},
    zip_safe=False,
    install_requires=[
        # "pytorch",    # caused a problem
        # "fairseq",
    ],
    python_requires=">=3.7",
)
