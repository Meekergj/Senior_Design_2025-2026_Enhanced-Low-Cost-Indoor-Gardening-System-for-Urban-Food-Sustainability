#from setuptools import setup
import setuptools as sut

# Note this will require a setuptools bdist_wheel or similar function later
sut.setup(
    name='Senior_Design_2025-2026_Indoor-Gardening-System',
    version='0.1',
    description='Program for for identifying the health of crops using AI imaging',
    author='README',
    author_email='README',
    packages=['Senior_Design_2025-2026_Indoor-Gardening-System'],
    install_requires=["tensorflow", "keras", "matplotlib"],
    license='MIT',
    python_requres=">=3.13.7",
)