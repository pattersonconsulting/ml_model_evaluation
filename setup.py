import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name='ml_valuation',
    version='0.0.1',
    author='Josh Patterson',
    author_email='josh@pattersonconsultingtn.com',
    description='Some good ol tools to analyze the monetory value of machine learning models',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/pattersonconsulting/ml_tools',
    project_urls = {
        "Bug Tracker": "https://github.com/pattersonconsulting/ml_tools/issues"
    },
    license='MIT',
    packages=['ml_valuation'],
    install_requires=['pandas', 'sklearn', 'matplotlib', 'numpy'],
)
