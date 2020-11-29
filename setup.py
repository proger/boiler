from setuptools import setup

setup(
    name='boiler',
    version='0.0.1',
    python_requires='>=3.8',
    author='Vlad Ki',
    author_email='vlad@kirillov.im',
    packages=['boiler'],
    long_description=open('README.md').read(),
    install_requires=[
        "annoy>=1.17,<1.18",
        "fastapi>=0.61.2,<0.62",
        "joblib>=0.18,<0.19",
        "librosa>=0.8,<0.9",
        "musicnn>=0.1,<0.2",
        "pydantic>=1.7,<1.8",
        "scipy>=1.5,<1.6.0",
        "torch>=1.7,<1.8.0",
        "torchaudio>=0.7,<0.8.0",
        "torchvision>=0.8,<0.9.0",
        "tqdm"
    ]
)

