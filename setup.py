import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="expert-score",
    version="0.0.1",
    author="Alireza Salemi",
    author_email="asalemi@cs.umass.edu",
    description="the implementation of the ExPerT score.",
    long_description=long_description,
    license_files=('LICENSE.txt',),
    long_description_content_type="text/markdown",
    url="https://github.com/alirezasalemi7/ExPerT",
    packages=setuptools.find_packages(
        include=['expert*'],  # ['*'] by default
        exclude=['expert.tests', 'expert.eval']
    ),
    install_requires=[
        "backoff==2.2.1",
        "json5==0.9.25",
        "openai==1.60.1",
        "parse==1.20.2",
        "protobuf==5.29.3",
        "setuptools==75.2.0",
        "tqdm==4.66.5",
        "transformers==4.45.2",
        "vllm==0.6.6",
        "google-generativeai==0.8.4",
    ],
)