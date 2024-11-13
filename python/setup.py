from setuptools import setup, find_packages

setup(
    name='your_package_name',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'black',
        'flake8',
        'ipython',
        'isort',
        'loguru',
        'matplotlib',
        'numpy',
        'pandas',
        'python-dotenv',
        'scikit-learn',
        'tqdm',
        'datargs',
        'lightning',
        'scikit-image',
        'torchvision',
        'albumentations',
        'wandb',
    ],
)
