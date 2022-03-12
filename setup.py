import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pidgin_ann_ben_fischer",
    version="0.0.1",
    author="Andy Wu",
    author_email="author@example.com",
    description="PIDGINv4 with Artificial Neural Network",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/andywuy/pidgin_ann",
    project_urls={
        "Bug Tracker": "https://github.com/andywuy/pidgin_ann/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: POSIX :: Linux",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.6",
)