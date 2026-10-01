from setuptools import find_packages, setup

setup(
    name="emotion-detector",
    version="0.1.0",
    description="Emotion detection application using the Skills Network Watson NLP EmotionPredict service",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "Flask>=3.0.0",
        "requests>=2.31.0",
    ],
)
