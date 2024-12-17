# Speckle Generator

A simple repo to generate a speckle pattern

For developing this code run `pip install -e .`

To download the package using pip: `pip install -i https://test.pypi.org/simple/ speckle-coolmule0`

To build and distribute this package, make sure `build` is up to date: `python3 -m pip install --upgrade build`. Then `python3 -m build` produces the file in the `dist/` folder. Upload it to the pypi testing server using python3 -m twine upload --repository testpypi dist/*, with the appropriate API key.