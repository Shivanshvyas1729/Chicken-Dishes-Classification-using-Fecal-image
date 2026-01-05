
import setuptools
from typing import List

try:
    with open("README.md", "r", encoding="utf-8") as f:
        long_description = f.read()
except FileNotFoundError:
    long_description = ""

__version__ = "0.0.0"

REPO_NAME = "Chicken-Dishes-Classification-using-Fecal-image"
AUTHOR_USER_NAME = "Shivanshvyas1729"
SRC_REPO = "cnnClassifier"
AUTHOR_EMAIL = "shivanshvyas1729@gmail.com"

def get_requirements(file_path: str) -> List[str]:
    requirements = []
    with open(file_path) as file:
        for line in file:
            line = line.strip()
            if not line or line.startswith('#') or line.startswith('-e'):
                continue
            requirements.append(line)
    return requirements

setuptools.setup(
    name=SRC_REPO,# must be pakage name
    version=__version__,
    author=AUTHOR_USER_NAME,
    author_email=AUTHOR_EMAIL,
    description="A small python package for ML application",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url=f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}",
    project_urls={
        "Bug Tracker": f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}/issues",
    },
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    install_requires=get_requirements("requirements.txt"),
)