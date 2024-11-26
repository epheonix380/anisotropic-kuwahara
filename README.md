# Kuwahara Processor Utility for Exr Files

Created to perform Anisotropic Kuwahara on EXR files

## Installation

Install pipenv using pip
`pip install pipenv`

Then in the root of the directory run
`pipenv install`

Invoke the virtualenv by using
`pipenv shell`

## Running GUI

In the virtual environment run this command:
`python main.py`
To open the GUI interface

## Running pytests

If you're on vs code you can run tests from the testing extension (much nicer experience).If you have no other choice then run it from the terminal. To run pytests from the terminal, you have to specify the python path. this will be the path to the anisotropic-kuwahara.

If you are on a git bash terminal:

```bash
export PYTHONPATH=$(pwd)

## alternatively:
# export PYTHONPATH=$PWD
```

Example commands to run tests:

```bash
pytest
pytest path/to/test/folder.py
pytest path/to/test/folder.py::test_name
```
