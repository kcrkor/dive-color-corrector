"""
This is a setup.py script generated by py2applet

Usage:
    python setup.py py2app
"""

from setuptools import find_packages, setup

APP = ['dcc.py']
DATA_FILES = []
OPTIONS = {
    'iconfile':'./logo/logo.icns'
}

setup(
    name="dive-color-corrector",
    version="1.2.0",
    packages=find_packages(),
    include_package_data=True,
    python_requires=">=3.11",
    install_requires=[
        "opencv-python>=4.8.0",
        "numpy>=1.24.0",
    ],
    extras_require={
        "gui": ["PySimpleGUI>=4.60.0"],
    },
    app=APP,
    data_files=DATA_FILES,
    options={'py2app': OPTIONS},
    setup_requires=['py2app'],
)
