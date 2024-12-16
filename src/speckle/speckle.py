import argparse
import random
import math


import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import scipy.fft


def white_balance(image) -> float:
	"""
	Calculatethe white/black balance of an image. 1=white. 0=black
	"""
	image_array = np.array(image)

	# Calculate the mean values for each channel (R, G, B)
	mean_rgb = np.mean(image_array, axis=(0, 1))

	# Assuming all grey, just take the max of any color channel
	max_mean = np.max(mean_rgb)

	return max_mean / 255


def speckle(image_width=500, image_height=500, circle_radius=5, circle_color = (0,0,0), desired_balance=0.5, variability=0):
	"""
	# A spaced pattern of circles
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
	image = Image.new("RGB", (image_width, image_height), "white")
	draw = ImageDraw.Draw(image)

	spacing = int(circle_radius * math.sqrt(math.pi / desired_balance))
	print(spacing)

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

def speckle_size(image, show=False) -> float:
	"""
	Calculates the average speckle size of an image using fourier transform
	"""
	image = image.convert("L")  # Convert image to grayscale
	image_array = np.array(image)

	# 2. Extract the central row and central column
	# Central row (if the image has an odd number of rows)
	central_row = image_array[image_array.shape[0] // 2, :]

	# 3. Compute the 1D FFT for the central row
	fft_central_row = np.fft.fft(central_row)

	fft_central_row_magnitude = np.abs(fft_central_row)

	# heights set to a large value to remove the noise
	peaks_info = scipy.signal.find_peaks(fft_central_row_magnitude, height=15000)

	freques = np.fft.fftfreq(central_row.size)
	# Get the actual peak locations from the data structure
	peaks = peaks_info[0]

	# Should have a +freqency and a matching -frequency peak only in the signal
	assert len(peaks) == 2
	
	peaks_freq = freques[peaks]
	# since the two frequencies match, just get one of them
	pix = 1/peaks_freq[0]

	if show:
		# Display the original image
		plt.subplot(1, 2, 1)
		plt.imshow(image_array, cmap='gray')
		plt.title("Original Grayscale Image")
		plt.axis("off")

		# Plot the FFT of the central row
		plt.subplot(1, 2, 2)
		plt.plot(np.fft.fftfreq(central_row.size), fft_central_row_magnitude)
		plt.title("FFT of Central Row")
		plt.xlabel("Frequency")
		plt.ylabel("Magnitude")

		plt.tight_layout()
		plt.show()

	return pix


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
					help ='black/white balance')
parser.add_argument('--output',
					metavar="O",
					type = str, 
					help ='output file name')
parser.add_argument('--ppi',
					required=False,
					type = float, 
					help ='image resolution in ppi, default is #todo')
parser.add_argument('--invert',
					required=False,
					action='store_true',
					help = 'show white pixels on black background rather than the other way round')
parser.add_argument('--seed',
					required=False,
					type = int,
					help = 'random number seed')
args = parser.parse_args()

parser.print_help()

random.seed(args.seed)

img = speckle(image_width = args.width, image_height = args.length, circle_radius=args.radius, desired_balance=args.bw_balance)
print(white_balance(img))
print(speckle_size(img, show=True))
img.save("img.png")
