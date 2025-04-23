from setuptools import find_packages, setup
from typing import List

def get_requirements(file_path:str)->List[str]:
    """
    This function returns the list of requirements
    """
    requirements=[]
    with open(file_path) as file_obj:
        requirements=file_obj.readlines()
        requirements=[req.replace('\n','') for req in requirements]
        if '-e .' in requirements:
            requirements.remove('-e .')
    return requirements

setup(
    name='Youtube Insight',
    version='0.1.0',
    author="Bayuzen Ahmad",
    author_email="bayuzen19@gmail.com",
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
)