import os
from setuptools import setup, find_packages

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name='disconnecting_framework',
    version='0.0.1',
    description='Framework for disconnecting tracklet graphs',
    author='David Karres',
    author_email='karres@physi.uni-heidelberg.de',
    packages=find_packages(include=['disconnecting_framework', 'disconnecting_framework.*']),
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/tkar-git/TrackletGraphs',
    entry_points={
        'console_scripts': [
            'tracklet = disconnecting_framework.core.click_script:cli'
        ]
    },
)