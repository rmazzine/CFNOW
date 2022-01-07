#!/usr/bin/env python
# -*- coding: utf-8 -*-
from setuptools import find_packages, setup

with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(
    name='cfnow',
    packages=find_packages('cfnow'),
    version='0.0.0',
    description='The easiest way to generate counterfactuals.',
    long_description='This algorithm uses a greedy strategy to find a counterfactual and then a Tabu search optimizer'
                     'to, possibly, find an optimal solution. This package works with any kind of model that outputs a '
                     'probability of a binary class. It also supports numerical, categorical binary and categorical '
                     'one-hot encoded features.',
    author='Raphael Mazzine Barbosa de Oliveira',
    author_email='mazzine.r@gmail.com',
    url='https://github.com/user/my-python-package',
    install_requires=required,
    license='MIT',
    keywords=['counterfactuals', 'counterfactual explanations', 'explainable artificial intelligence', 'xai', 'cf'],
    classifiers=[
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
    ],
)