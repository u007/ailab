from PIL import Image, ImageChops
import PIL.ImageOps    

image = Image.open('white+png.png')
inverted_image = ImageChops.invert(image)
# gray_image = image.convert('L')
# inverted_image = PIL.ImageOps.invert(gray_image)

inverted_image.save('white+png-inverted.png')
