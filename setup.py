from setuptools import setup, find_packages
from typing import List

HYPHEN_E_DOT = "-e ."

def get_requirements() -> List[str]:
    with open('requirements.txt', 'r') as f:
        return [x for x in f.read().split('\n') if x != HYPHEN_E_DOT]

setup(
    name='diabetes diagnose',
    version='0.1',
    author='Jaber Rahimifard',
    author_email='jaber.rahimifard@outlook.com',
    packages=find_packages(),
    install_requires=get_requirements()
)
