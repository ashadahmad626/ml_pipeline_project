from setuptools import find_packages, setup
from typing import List

HYPHEN_E_DOT = "-e ."

def get_requirements(filepath: str) -> List[str]:
    requirements = []
    with open(filepath) as file_obj:
        requirements = file_obj.readlines()
        requirements = [i.replace("\n", "") for i in requirements]
        if HYPHEN_E_DOT in requirements:
            requirements.remove(HYPHEN_E_DOT)
    return requirements

setup(
    name='ML_Pipeline_project',
    version='1.0.0',
    description='Production-Ready Machine Learning Pipeline — Income Prediction',
    author='Ashad Ahmad',
    author_email='ahmadashad0786@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements("requirements.txt"),
    python_requires=">=3.9",
)