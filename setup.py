from setuptools import setup

setup(
    name='boiler',
    version='0.0.1',
    author='Vlad Ki',
    author_email='vlad@kirillov.im',
    packages=['boiler'],
    long_description=open('README.md').read(),
    install_requires=[
        "annoy",
        "librosa",
        "scipy",
        "torch",
        "torchaudio",
        "torchvision",
        "tqdm"
    ]
)

