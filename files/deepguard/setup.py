from setuptools import setup, find_packages

setup(
    name="deepguard",
    version="1.0.0",
    description="AI-Generated Image Detector for cybersecurity professionals",
    author="Your Name",
    python_requires=">=3.8",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.21",
        "Pillow>=9.0",
        "opencv-python>=4.5",
    ],
    entry_points={
        "console_scripts": [
            "deepguard=deepguard.cli:main",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Topic :: Security",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
