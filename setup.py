from setuptools import setup, find_packages

setup(
    name='chainai',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'litellm',
        'requests',
        'numpy'
    ],
    description='A chain AI model for generating prompts and handling outputs',
    author='Ben Baptist',
    author_email='sawham6@gmail.com',
    url='https://github.com/benbaptist/chainai',  
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)