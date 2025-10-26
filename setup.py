"""
Setup script for Noeta DSL
"""
from setuptools import setup, find_packages

setup(
    name='noeta',
    version='0.1.0',
    description='A Domain-Specific Language for data analysis that compiles to Python/Pandas code',
    author='Noeta Team',
    python_requires='>=3.8',
    py_modules=[
        'noeta_lexer',
        'noeta_parser',
        'noeta_ast',
        'noeta_codegen',
        'noeta_runner',
        'noeta_kernel',
        'install_kernel',
        'test_noeta'
    ],
    install_requires=[
        'pandas>=1.3.0',
        'numpy>=1.20.0',
        'matplotlib>=3.3.0',
        'seaborn>=0.11.0',
        'scipy>=1.7.0',
        'scikit-learn>=0.24.0',
        'jupyter>=1.0.0',
        'ipykernel>=6.0.0',
    ],
    entry_points={
        'console_scripts': [
            'noeta=noeta_runner:main',
        ],
    },
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Compilers',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
    ],
)
