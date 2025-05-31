from setuptools import setup, find_packages

# Read requirements from requirements.txt
with open("requirements.txt") as f:
    required = f.read().splitlines()

setup(
    name="msna-sim",
    version="0.1.0",
    description="A library for generating realistic synthetic MSNA signals with physiologically accurate characteristics.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Ryan 'RyanIRL' Peters",
    author_email="ryanirl@icloud.com",
    url="https://github.com/ryanirl/msna-sim",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Healthcare Industry",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=required,
    entry_points={
        "console_scripts": [
            "msna-sim-dashboard=msna_sim.dashboard:main_cli"
        ],
    },
    keywords="msna, nerve activity, signal simulation, synthetic data, microneurography, cardiovascular, autonomic nervous system",
    project_urls={
        "Bug Reports": "https://github.com/ryanirl/msna-sim/issues",
        "Source": "https://github.com/ryanirl/msna-sim",
        "Documentation": "https://github.com/ryanirl/msna-sim/blob/main/SPECS.md",
    },
)