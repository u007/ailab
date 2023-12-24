import openai
import requests
from PIL import Image
from io import BytesIO

# URL of the uploaded couple photo
photo_url = 'input.jpg'

# Setting up the prompt for a studio photograph
# damn openai
prompt = f"Create a studio quality photograph based on the couple in the image: {photo_url} as traditional Chinese dress.\
  Persons in the photo should look like the twin of both couple in the photograph, not cartoons or illustrations. Top half and bottom half of the persons in the photograph must be\
  visible. Ensure the face and hand is not mutilated."

# Sending the request to DALL-E 3
try:
    response = openai.Image.create(
        prompt=prompt,
        n=1,
        size="512x512"
    )
    print("Image generated successfully.")

    # The URL of the generated image
    image_url = response['data'][0]['url']

    # Downloading the image
    image_response = requests.get(image_url)
    image = Image.open(BytesIO(image_response.content))
    image.save('output.png')
    print("Image saved as output.png.")

except Exception as e:
    print(f"An error occurred: {e}")
