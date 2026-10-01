from setuptools import find_packages, setup

setup(
    name="emotion-detector",
    version="0.1.0",
    description="Emotion detection application using IBM Watson NLP",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "Flask>=3.0.0",
        "ibm-watson>=10.0.0",
    ],
)
