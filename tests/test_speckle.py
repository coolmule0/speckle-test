from PIL import Image
from speckle import speckle

def test_image_size():
    """
    function arguments create correct size image
    """
    image = speckle.speckle(image_width=20, image_height=20)
    width, height = image.size
    assert width == 20
    assert height == 20

def test_speckle_size():
    """
    A test speckle image calculates the correct pixel size using a fourier transform
    """
    assert False

def test_white_balance():
    """Computes the white balance for a single circle centred on (6,6)
    """
    image = speckle.speckle(image_width=13, image_height=13,  circle_radius=5)
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

def test_image_comparison_2():
    """And that the opposite is false - different radius
    """
    image = speckle.speckle(image_width=200, image_height=200,  circle_radius=4, variability=0)
    test_image = Image.open("tests/test_grid.png")

    pixels1 = list(image.getdata())
    pixels2 = list(test_image.getdata())

    assert pixels1 != pixels2