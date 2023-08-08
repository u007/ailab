import keras_ocr
import matplotlib.pyplot as plt

image_path = "../../compassrose/public_html/images/compass-rose-lg2b.jpg"
# Create pipeline 
pipeline = keras_ocr.pipeline.Pipeline()

img = keras_ocr.tools.read(image_path)

# Get predictions
prediction_groups = pipeline.recognize([img])

# Draw boxes and annotations on image
annotated_img = keras_ocr.tools.drawAnnotations(image=img, predictions=prediction_groups[0])

# Save annotated image 
plt.imshow(annotated_img)
plt.axis('off')
plt.savefig(image_path+'-clean.png', bbox_inches='tight', pad_inches=0)

# img = keras_ocr.tools.read(image_path)# Prediction_groups is a list of (word, box) tuples
# prediction_groups = pipeline.recognize([img])#print image with annotation and boxes
# keras_ocr.tools.drawAnnotations(image=img, predictions=prediction_groups[0])

# # Save image
# plt.imshow(img)
# plt.savefig(image_path+'-empty.png')
