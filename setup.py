from setuptools import setup, find_packages

setup(
    name="accelerated_features",
    version="0.1",
    description="Accelerated Features Submodule",
    packages=find_packages(where="accelerated_features"),
    package_dir={"": "accelerated_features"},
    include_package_data=True,
)