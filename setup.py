from setuptools import setup, find_packages

# Read the README file for the long description
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Define the setup configuration
setup(
    name="coded_wfs_sim",
    version="0.1.0",
    description="A Python package for simulating light propagation through 3D structures.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Syed Kazim",
    author_email="syed.kazim@uni-siegen.de",
    url="https://github.com/Muhammad-Kazim/coded_wfs_sim",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
    install_requires=[],  # Dependencies are managed via environment.yml
    include_package_data=True,  # Include additional files specified in MANIFEST.in
    zip_safe=False,  # To avoid issues with certain editable installs
)
