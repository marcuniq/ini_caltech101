from setuptools import setup
from setuptools import find_packages

setup(name='ini_caltech101',
      version='0.0.1',
      description='',
      author='Marco Unternaehrer',
      author_email='marco.unter@gmail.com',
      install_requires=['keras==0.2.0', 'scikit-learn', 'Image'],
      packages=find_packages())
