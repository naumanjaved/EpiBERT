import setuptools
from pathlib import Path

# --- Package Information ---
NAME = "epibert"
VERSION = "1.1.0"
DESCRIPTION = "Predicting CAGE-seq from ATAC-seq and DNA sequence."
URL = "https://github.com/naumanjaved/EpiBERT"
AUTHOR = "N Javed"
AUTHOR_EMAIL = "javed@broadinstitute.org"

# --- Setup Configuration ---
here = Path(__file__).resolve().parent

with open(here / "README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

def get_install_requires():
    """Reads requirements.txt and preprocesses it for setuptools."""
    requirements = []
    req_path = here / "requirements.txt"
    if not req_path.exists():
        req_path = Path("requirements.txt")
        
    with open(req_path, "r") as f:
        for line in f.read().splitlines():
            if line and not line.strip().startswith('#'):
                requirements.append(line)
    return requirements

setuptools.setup(
    name=NAME,
    version=VERSION,
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type="text/markdown",
    url=URL,
    project_urls={"Bug Tracker": f"{URL}/issues"},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.8",
    install_requires=get_install_requires(),
    include_package_data=True,
)