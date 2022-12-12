"""Setup for the reinforcement learning package"""
from setuptools import setup

setup(
   name='rlpackage',
   version='0.0',
   description='Rl package for world model',
   author='Guillaume LEVY',
   author_email='levyguillaume10@yahoo.fr',
   packages=['rlpackage'],  # would be the same as name
   install_requires=['wheel', 'bar'], #external packages acting as dependencies
)