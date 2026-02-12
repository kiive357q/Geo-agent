from setuptools import setup, find_packages

with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='geo-agent',
    version='1.0.0',
    description='智能桩基检测 Geo-Agent 系统',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Engineering Team',
    author_email='contact@geo-agent.com',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'torch>=2.0.0',
        'numpy>=1.24.0',
        'scipy>=1.10.0',
        'fastapi>=0.100.0',
        'uvicorn>=0.23.0',
        'pydantic>=2.0.0',
        'python-multipart>=0.0.6',
        'requests>=2.31.0',
        'matplotlib>=3.7.0',
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'License :: Private :: Do Not Distribute',
        'Operating System :: OS Independent',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Engineering',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    python_requires='>=3.10',
    entry_points={
        'console_scripts': [
            'geo-agent=main:main',
        ],
    },
)
