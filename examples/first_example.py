from speckle import speckle

def example_1() -> None:
	"""
	Generate a simple 500x500 speckle pattern with a tiny amount of randomness
	"""
	print("Generating a 500 by 500 speckle pattern to img.png")
	image = speckle.speckle(image_width=500, image_height=500, circle_radius=5, circle_color = (0,0,0), desired_balance=0.5, variability=0.05)
	
	print("Display the image generated and fourier transform")
	speckle.speckle_size(image, show=True)
	print("Done!")

if __name__ == "__main__":
	example_1()
