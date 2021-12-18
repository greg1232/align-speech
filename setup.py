import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="align-speech",
    version="0.0.1",
    author="Greg Diamos",
    author_email="gregory.diamos@gmail.com",
    description="A simple library for forced alignment.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/greg1232/speech-align",
    project_urls={
        "Bug Tracker": "https://github.com/greg1232/speech-align/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache2 License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "align_speech"},
    packages=setuptools.find_packages(where="align_speech"),
    python_requires=">=3.6",
)
