import os

from ini_caltech101.dataset import caltech101

# width and height of resized images
width, height = 240, 180


print("Resizing images...")
input_dir = os.path.abspath(os.path.join('datasets', 'original', '101_ObjectCategories'))
output_dir = os.path.abspath(os.path.join('datasets', 'original-resized', '101_ObjectCategories'))
path = caltech101.resize_imgs(input_dir, output_dir, width=width, height=height)

print('path: ', path)
