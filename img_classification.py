import keras
from PIL import Image, ImageOps
import numpy as np


def teachable_machine_classification(img, weights_file):
    # Load the model
    model = keras.models.load_model(weights_file)

    # Create the array of the right shape to feed into the keras model
    data = np.ndarray(shape=(1, 28, 28, 3), dtype=np.float32)
    image = img
    # image sizing
    size = (28, 28)
    image = ImageOps.fit(image, size, Image.ANTIALIAS)

    # turn the image into a numpy array
    image_array = np.asarray(image)
    # Normalize the image
    # normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1

    # Load the image into the array
    if data[0].shape != image_array.shape:
        image2 = np.expand_dims(image_array, -1)
        x1 = image2.shape[0]
        x2 = image2.shape[1]
        image2 = np.reshape(np.broadcast_to(image2, (x1, x2, 3)), (x1, x2, 3))
        data[0] = image2
    else:
        data[0] = image_array

    # run the inference
    prediction = model.predict(data)

    return np.argmax(prediction)  # return position of the highest probability
