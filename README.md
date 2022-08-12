# Aerial-images-to-maps

## Aim:
This thesis explores the structure of a cyclic generative adversarial system network, as well as to see if the model can provide effective results when applied to test images. Throughout this section, you will find the code that was used to complete the relevant tasks.

### Example Images

- Aerial image

![example image aerial.jpg](https://github.com/Astrojigs/Aerial-images-to-maps/blob/main/Images/example%20image%20aerial%20-%20Copy.jpg)
- Map like representation

![example image map](https://github.com/Astrojigs/Aerial-images-to-maps/blob/main/Images/example%20image%20map%20-%20Copy.jpg)

The data is taken from [Kaggle](https://www.kaggle.com/datasets/suyashdamle/cyclegan).

# Loss functions:
Following loss functions are used to train the model:

- Generator Loss
- Discriminator Loss
- Cyclic Loss
- Identity Loss

They all can be found [here](https://github.com/Astrojigs/Aerial-images-to-maps/blob/main/losses.py)

# Models:

Models were trained on *128x128* pixels and *256x256* pixels.

## Code:
**Generator architectures** can be found here: [link](https://github.com/Astrojigs/Aerial-images-to-maps/blob/main/Generator_arc.py)

**Discriminator architectures** can be found here: [link](https://github.com/Astrojigs/Aerial-images-to-maps/blob/main/Discriminator_arc.py)

[Model architecture plots](https://github.com/Astrojigs/Aerial-images-to-maps/tree/main/Images/architectures)
