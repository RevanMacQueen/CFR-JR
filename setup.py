import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="cfr_jr",
    version="0.0.1",
    author="Revan MacQueen",
    author_email="revan@ualberta.ca",
    description="git repo for an implementation of CFR-JR",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/RevanMacQueen/CFR-JR",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "cfr_jr"},
    packages=setuptools.find_packages(where="cfr_jr"),
    python_requires=">=3.7",
)