import sys
import pytest
import os
from unittest.mock import patch
from PIL import Image
import numpy as np
import matplotlib
from speckle import speckle

# remove matplotlib from showing
matplotlib.use("Agg")


def test_image_size():
	"""
	function arguments create correct size image
	"""
	image = speckle.speckle(image_width=20, image_height=20, variability=0)
	width, height = image.size
	assert width == 20
	assert height == 20

def test_white_balance():
	"""Computes the white balance for a single circle centred on (6,6)
	"""
	image = speckle.speckle(image_width=13, image_height=13,  circle_radius=5, variability=0)
	# 42% of the image is speckle pixels
	assert int(speckle.white_balance(image) * 100) == 42

def test_image_comparison():
	"""Compares the generated speckle patern to a pregenerated image
	"""
	image = speckle.speckle(image_width=200, image_height=200,  circle_radius=5, variability=0)
	test_image = Image.open("tests/test_grid.png")

	pixels1 = list(image.getdata())
	pixels2 = list(test_image.getdata())

	assert pixels1 == pixels2

	"""And that the opposite is false - different radius
	"""
	image = speckle.speckle(image_width=200, image_height=200,  circle_radius=4, variability=0)
	pixels3 = list(image.getdata())

	assert pixels3 != pixels2

def test_white_balance():
	# Test all background (white image)
	white_image = Image.new("RGB", (100, 100), (255, 255, 255))
	assert speckle.white_balance(white_image) == pytest.approx(1.0)

	# Test all speckle (black image)
	black_image = Image.new("RGB", (100, 100), (0, 0, 0))
	assert speckle.white_balance(black_image) == pytest.approx(0.0)

	# Computes the white balance for a single circle centred on (6,6)
	single_image = speckle.speckle(image_width=13, image_height=13,  circle_radius=5, variability=0)
	# Compare against the pre-computed value
	assert speckle.white_balance(single_image) == pytest.approx(0.4260355)


def test_speckle():
	# Generate speckle with 0 balance
	img = speckle.speckle(image_width=100, image_height=100, desired_balance=0.0)
	# correct size from arguments
	assert img.size == (100, 100)
	# entirely white result
	assert speckle.white_balance(img) == pytest.approx(1.0)

	# Test with 1 balance
	img = speckle.speckle(image_width=100, image_height=100, desired_balance=1.0)
	assert speckle.white_balance(img) < 0.1 #only approximate method

def test_argparse_missing_required_arguments():
	test_args = [
		"speckle_generator.py",
		"-w", "500",
		"-l", "500"
	]
	with patch.object(sys, "argv", test_args), pytest.raises(SystemExit):
		speckle.parse_args()  # Missing required `--radius` and `--bw_balance`

@pytest.fixture(autouse=True)
def test_end_to_end():
	"Generates a correct image"
	test_args = [
        "speckle_generator.py",
        "-w", "400",
        "-l", "500",
        "-r", "10",
        "-bw", "0.5",
        "-i", "png",
		"--output", "test"
    ]
	with patch.object(sys, "argv", test_args):
		speckle.main()
		assert os.path.exists("test.png")

		image = Image.open("test.png")
		w, l = image.size
		assert w == 400
		assert l == 500