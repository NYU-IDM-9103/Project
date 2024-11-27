## Data

Observations: CIFAR10 dataset is quite easy to load as part of a library. Easy to use as well. Main problem would be the low resolution (32x32). Would be a good data source to quickly test out things but would eventually work with higher resolution image. 

## Pre-trained model

Observations: tried loading and using AlexNet. Was pretty straightforward in returning classification labels with the highest probabilities. But not sure how to manipulate model params since the model is already trained? Might not be able to reverse engineer and play with the trained model params?

Per this [example](https://machinelearningmastery.com/how-to-visualize-filters-and-feature-maps-in-convolutional-neural-networks/), I can visualize feature maps from very specific layers in the model. So I could use a pre-trained model, and maybe instead of modifying model parameters, I can introduce noise to the input of the intermediate layers and see how it looks?
