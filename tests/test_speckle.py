from speckle import speckle

def test_image_size():
    """
    function arguments create correct size image
    """
    image = speckle.speckle(image_width=20, image_height=20)
    width, height = image.size
    assert width == 20
    assert height == 20
