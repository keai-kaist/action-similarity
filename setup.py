from setuptools import setup, find_packages

setup(
    name='action_similarity',
    version='1.0.0',
    author='Simcs',
    author_email='smh3946@gmail.com',
    url='https://github.com/keai-kaist/action-similarity',
    classifiers=[
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3 :: Only'
    ],
    packages=find_packages(),
    install_requires=[
        'numpy',
        'scipy',
        'tqdm',
        'glob',
        'more-itertools',
        'tslearn',
        'opencv-python',
    ],
    python_requires='>=3.8',
)