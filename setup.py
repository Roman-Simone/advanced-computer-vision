import os
from setuptools import setup, find_packages

setup(
    name="xfeat",
    version="1.0",
    description="XFeat Package for Feature Matching",
    author="OpenAI & ZeyiSun",
    packages=find_packages(include=["xfeat*", "modules*"]),
    package_dir={"": "."},
    install_requires=[],
    include_package_data=True,
    extras_require={"dev": ["pytest"]},
)
