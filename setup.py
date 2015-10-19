from setuptools import setup
from setuptools import find_packages

setup(name='ini_caltech101',
      version='0.0.1',
      description='',
      author='Marco Unternährer',
      author_email='marco.unter@gmail.com',
      install_requires=['keras', 'python-resize-image', 'Image'],
      packages=find_packages())
