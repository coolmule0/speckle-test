import argparse
import random
import math
import time


import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import scipy.fft


def white_balance(image) -> float:
	"""
	Calculate the white/black balance of an image. 1=white. 0=black
	"""
	image_array = np.array(image)

	# Calculate the mean values for each channel (R, G, B)
	mean_rgb = np.mean(image_array, axis=(0, 1))

	# Assuming all grey, just take the max of any color channel
	max_mean = np.max(mean_rgb)

	return max_mean / 255


def speckle(image_width=500, image_height=500, circle_radius=5, circle_color = (0,0,0), background=(255,255,255), desired_balance=0.5, variability=0):
	"""
	# Generate an image of a spaced pattern of circles
	# User-defined parameters
	# image_width = 500         # Width of the image
	# image_height = 500        # Height of the image
	# circle_radius = 5        # Radius of the circles
	# # spacing = 60              # Spacing between circle centers
	# circle_color = (0, 0, 0)  # Color of the filled circles (black in this case)
	# desired_balance = 0.5       # how much white/black balance in the image
	# variability = 0.5 # between 0-1. 0 = uniform 1= up to 2 boxes away
	"""
	# Create a blank white image
	image = Image.new("RGB", (image_width, image_height), background)
	draw = ImageDraw.Draw(image)

	spacing = int(circle_radius * math.sqrt(math.pi / desired_balance))
	# print(spacing)

	# Calculate the number of circles in both x and y directions
	num_circles_x = image_width // spacing
	num_circles_y = image_height // spacing

	# Draw circles at regular intervals
	for i in range(-1,num_circles_x+1):
		for j in range(-1,num_circles_y+1):
			# Calculate the center of each circle
			center_x = i * spacing + spacing // 2
			center_y = j * spacing + spacing // 2

			offset_x = random.randint(int(-spacing*variability/2), int(spacing*variability/2))
			offset_y = random.randint(int(-spacing*variability/2), int(spacing*variability/2))
			center_x += offset_x
			center_y += offset_y

			# Draw the circle (outline)
			left_up = (center_x - circle_radius, center_y - circle_radius)
			right_down = (center_x + circle_radius, center_y + circle_radius)
			draw.ellipse([left_up, right_down], fill=circle_color)
	return image

def speckle_fft(image, show=True):
	image_f = image.convert('L')  # Convert to grayscale ('L' mode)

	# Convert PIL image to a numpy array
	image_array = np.array(image_f)

	fft_image = scipy.fft.fft2(image_array)

	# Shift the zero-frequency component to the center
	fft_image_shifted = scipy.fft.fftshift(fft_image)

	# Compute the magnitude spectrum 
	magnitude_spectrum = np.abs(fft_image_shifted)
	# log it
	magnitude_spectrum_log = np.log(magnitude_spectrum + 1)

	# use the central (horizontal) row of the 2d FFT
	central_row = magnitude_spectrum[magnitude_spectrum_log.shape[0] // 2]

	# frequency plot is symetic, so remove half
	half_ind = central_row.shape[0] // 2 + 1

	if show:
		# Display the original image
		plt.subplot(1, 2, 1)
		plt.imshow(image_array, cmap='gray')
		plt.title("Original Grayscale Image")
		# plt.axis("off")

		# # Create frequency axes
		M, N = image_array.shape
		freq_x = np.fft.fftfreq(N, 1)  # Horizontal frequency axis
		freq_x = np.fft.fftshift(freq_x)  # Shift the zero-frequency component to the center

		plt.subplot(1, 2, 2)
		plt.plot(freq_x[half_ind:], central_row[half_ind:])
		plt.title("Magnitude FFT Spectrum of Speckles")
		plt.xlabel('Horizontal Frequency')
		plt.ylabel('Magitude')

		plt.tight_layout()
		plt.show()

def parse_args():
	parser = argparse.ArgumentParser(description ='speckle generator')
	parser.add_argument('-w', '--width', metavar ='W', 
						type = int,
						required=True,
						help ='width of the image')
	parser.add_argument('-l', '--length', metavar ='H', 
						type = int,
						required=True,
						help ='length of the image')
	parser.add_argument('-r', '--radius', metavar ='R', 
						type = int,
						required=True,
						help ='radius of the speckle circle')
	parser.add_argument('-bw', '--bw_balance',
						metavar="[0.0-1.0]",
						type = float,
						required=True,
						help ='black/white balance. 0 is no speckles. 1 is entirely speckles')
	parser.add_argument('--output',
						metavar="O",
						type = str, 
						help ='output file name')
	parser.add_argument('--dpi',
						required=False,
						type = int,
						default=300,
						help ='image resolution in pixels per inch, default is 300')
	parser.add_argument('--invert',
						required=False,
						action='store_true',
						help = 'show white pixels on black background rather than the other way round')
	parser.add_argument('--seed',
						required=False,
						type = int,
						help = 'random number seed. Timed-random if unset')
	return parser.parse_args()


if __name__ == "__main__":
	args = parse_args()

	# Initialize seed random
	if args.seed:
		random.seed(args.seed)
	else:
		random.seed(time.time())
	
	# Set figure colours
	speckle_c = (0,0,0)
	background_c = (255,255,255)
	if args.invert:
		speckle_c = (255,255,255)
		background_c = (0,0,0)

	img = speckle(image_width = args.width, image_height = args.length, circle_radius=args.radius, desired_balance=args.bw_balance, circle_color=speckle_c, background=background_c)
	print("The amount of white balance is ", white_balance(img))
	
	speckle_fft(img)

	print("saving pattern to img.png")
	img.save("img.png", dpi=(args.dpi, args.dpi))

