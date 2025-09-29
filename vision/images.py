import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import cv2

def process_image(image_path):
    """
    Loads an image, converts it to grayscale, resizes it to 64x64,
    and normalizes the pixel values.
    """
    image = mpimg.imread(image_path)

    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    resized_image = cv2.resize(gray_image, (128, 128))

    normalized_image = resized_image / 255.0

    return normalized_image

# Example usage:
image_path = "/Users/nicolasalan/Documents/stacks/vision/WhatsApp Image 2025-09-12 at 17.32.03.jpeg"
processed_image = process_image(image_path)

# Save the processed image
output_path = "/Users/nicolasalan/Documents/stacks/vision/processed_image.png"
plt.imsave(output_path, processed_image, cmap='gray')

# Display the processed image
plt.imshow(processed_image, cmap='gray')
plt.show()

print(f"Image saved to {output_path}")
