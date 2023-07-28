# Importing Required Modules
from rembg import remove
from PIL import Image

def convert_to_png(file: str):
  im1 = Image.open(file)
  # save file with extension png with original file name
  #   also must support multiple dot
  new_name = file.split('.')[0] + '.png'
  im1.save(new_name)
  # Store path of the image in the variable input_path
  return new_name

input_file = 'compass-rose-h.jpg'
# png_file = convert_to_png(input_file)
# Store path of the output image in the variable output_path
output_path = 'compass-rose-h.clean2.png'
  
# Processing the image
input = Image.open(input_file)
  
# Removing the background from the given Image
output = remove(input)
  
#Saving the image in the given path
output.save(output_path)