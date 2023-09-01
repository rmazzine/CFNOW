from setuptools import setup, find_packages

README = open('README.md').read()

HISTORY = open('HISTORY.md').read()

REQUIREMENTS = [r.replace('\n', '') for r in open('requirements.txt').readlines()]

setup_args = {
    'name': 'CFNOW',
    'version': '0.0.1',
    'description': 'Generate counterfactuals with ease. This package takes a model and point (with a certain class) '
                   'and minimally changes it to flip the classification result.',
    'long_description_content_type': 'text/markdown',
    'long_description': f'{README}\n{HISTORY}',
    'license': 'MIT',
    'package_dir': {'cfnow':''},
    'packages': find_packages(),
    'author': 'Raphael Mazzine Barbosa de Oliveira',
    'author_email': 'mazzine.r@gmail.com',
    'url': '',
    'keywords': [
        'counterfactuals',
        'counterfactual explanations',
        'flipping original class',
        'explainable artificial intelligence'
    ],
    'classifiers': [
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
    ]
}

install_requires = REQUIREMENTS

if __name__ == '__main__':
    setup(**setup_args, install_requires=install_requires)
