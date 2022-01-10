from distutils.core import setup
from setuptools import find_packages

setup(
    name='RNN_Adaptation',
    packages=find_packages(),
    version='0.0.1',
    description='Fast model adaptation for system identification with deep learning',
    keywords=["control", "transfer learning", "system identification"],
    include_package_data=True,
    python_requires='>=3.8.*',
    classifiers=[
          'Natural Language :: English',
          'Programming Language :: Python :: 3.8',
    ],
    author='Forgione, M. and Muni, A. and Gallieri, M. and Piga, D.',
    author_email='marco.forgione@supsi.ch',
    url='',
    requires=[],
    zip_safe=True,
    license='MIT'
)

# EOF
