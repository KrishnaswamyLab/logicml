from setuptools import setup
setup(
    name='logicml',
    version='1.0',
    description='logicml -- experimental code for the intersection of machine learning and logic circuits',
    url='https://github.com/schlaumli/logicml',
    author='Tobias Brudermueller',
    author_email='tobias.brudermueller@rwth-aachen.de',
    license='GPL-3',
    packages=['logicml'],
    long_description=open('README.md').read(),
    zip_safe=False
)