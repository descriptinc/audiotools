from setuptools import setup, find_packages

with open('README.md') as f:
    long_description = f.read()

setup(
    name='audiotools',
    version="0.0.1",
    classifiers=[
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.7',
        'Topic :: Artistic Software',
        'Topic :: Multimedia',
        'Topic :: Multimedia :: Sound/Audio',
        'Topic :: Multimedia :: Sound/Audio :: Editors',
        'Topic :: Software Development :: Libraries',
    ],
    description='Utilities for handling audio.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Prem Seetharaman',
    author_email='prem@descript.com',
    license='MIT',
    packages=find_packages(),
    install_requires=[
        'soundfile',
        'pyloudnorm',
        'scipy',
        'torch',
        'julius',
        'torchaudio',
        'ffmpy',
        'ipython',
        'rich',
        'matplotlib',
        'librosa'
    ],
    extras_require={
        'tests': ['pytest', 'pytest-cov', "line_profiler", "tqdm", "argbind"],
    }
)