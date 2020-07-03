from setuptools import setup, find_packages


def _requires_from_file(filename):
    return open(filename).read().splitlines()


setup(
    name="colony_analysis",
    version='1.0',
    description='colony_analysis is xxx',
    author='Takeshi Nomura',
    author_email='nomura.takeshi.no3@is.naist.jp',
    install_requires=_requires_from_file('requirements.txt'),
    packages=find_packages()
)
