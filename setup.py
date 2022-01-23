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
    url="https://github.com/greg1232/align-speech",
    project_urls={
        "Bug Tracker": "https://github.com/greg1232/speech-align/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache2 License",
        "Operating System :: OS Independent",
    ],
    packages=setuptools.find_packages(where="."),
    package_data  = {
            'align_speech': ['configs/peoples_speech_dev_set.yaml'],
    },
    install_requires = [
        "python-configuration[yaml]",
        "numpy",
        "gruut",
        "srt",
        "google-cloud-speech",
        "smart_open[gcs,s3]",
        "pydub",
        "alignment"
    ],
    python_requires=">=3.6",
)
