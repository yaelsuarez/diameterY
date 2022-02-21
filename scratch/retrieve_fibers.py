from diametery.fiber import Image
import matplotlib.pyplot as plt
import numpy as np
from diametery.skeleton import get_total_flux

image = Image.create()
im = image.render_image()
plt.imshow(im)

pred = image.render_field_and_weights()[:,:,0:2 ]
total_flux = get_total_flux(pred)
plt.imshow(total_flux)
plt.show()