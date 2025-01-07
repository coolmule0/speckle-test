**Speckle Generator** 
===================== 

**Overview**
------------
 
The Speckle Generator script creates speckle patterns with configurable properties, computes their Fast Fourier Transform (FFT) for frequency analysis, and saves the generated image. This tool is a simple practice test case and not actually suitable for proper speckle generation. 

![Example of a speckle pattern](img.png)

**Features** 
------------ 

	1. **Speckle Pattern Generation** 
		* Configurable image dimensions, speckle size, and distribution. 
		* Adjustable speckle-to-background balance. 
		* Variability for randomized speckle placement. 
	2. **White Balance Calculation** 
		* Computes the proportion of background visible in the image. 
	3. **FFT Visualization** 
		* Computes and visualizes the frequency spectrum of the speckle pattern. 
	4. **Customizable Outputs** 
		* Save the generated image in various formats (e.g., PNG, TIFF, BMP). 
		* Option to invert the color scheme for enhanced visualization. 

**Requirements**
---------------- 
	
	* Python 3.13 or later 

Install the required libraries using the pypi test environmnet `pip install -i https://test.pypi.org/simple/speckle-coolmule0`, or locally using: `pip install`.


**Development**
---------------- 

For developing the code run `pip install -e .`

To build and distribute this package, make sure `build` is up to date: `python3 -m pip install --upgrade build`. Then `python3 -m build` produces the file in the `dist/` folder. Upload it to the pypi testing server using python3 -m twine upload --repository testpypi dist/*, with the appropriate API key.

Testing can be carried out with `pytest tests`.

**Usage**
---------

Run the script from the command line with the required arguments: 

### **Basic Command**

`python src/speckle/speckle_generator.py -w WIDTH -l LENGTH -r RADIUS -bw BW_BALANCE -i IMAGE_FORMAT`

### **Arguments**

Help with running the function can be found with the `-h` argument for help: `python src/speckle/speckle_generator.py -h`

| Argument | Type | Description | Required |
| --- | --- | --- | --- | 
| `-w, --width` | `int` | Width of the image in pixels. | Yes |
| `-l, --length` | `int` | Length of the image in pixels. | Yes |
| `-r, --radius` | `int` | Radius of the speckle circles in pixels. | Yes |
| `-bw, --bw_balance` | `float` | Black/white balance, between `0.0` (no speckles) and `1.0` (entirely speckles). | Yes |
| `-i, --image_format` | `str` | Output image format. Must be one of `png`, `tiff`, or `bmp`. | Yes |
| `--output` | `str` | Name of the output file (default is `img.png`). | No | | `--dpi` | `int` | Resolution of the output image in DPI (default: `300`). | No |
| `--invert` | `flag` | Invert the color scheme (white speckles on black background). | No |
| `--seed` | `int` | Random seed for consistent speckle patterns. If not set, uses a time-based seed. | No | 


**License**
----------- 

This script is open-source and available under the GNU GPL License. 

**Contributing**
---------------- 

Contributions are welcome! Feel free to submit issues or pull requests.