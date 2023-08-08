from super_image import EdsrModel, ImageLoader
from PIL import Image
import requests

path = 'compass-rose-lg2b_cleanup.jpg'
# path = '/Users/james/Downloads/4010be9a-9add-460f-b3f9-db2ec91bd167.png'
url = r'compass-rose-lg2b_cleanup.jpg'
image = Image.open(url)

model = EdsrModel.from_pretrained('eugenesiow/edsr-base', scale=2)
inputs = ImageLoader.load_image(image)
preds = model(inputs)

ImageLoader.save_image(preds, path + '-scaled.png')# './IMG_2687-scaled.png')
# ImageLoader.save_compare(inputs, preds, './IMG_2687-compared.png')
