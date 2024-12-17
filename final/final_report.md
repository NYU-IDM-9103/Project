## Intro and Motivations

The original idea I wanted to work with was to critique the obsession with perfection and optimization in technology, especially in ML algorithms, which is usually related to either some form of accuracy or efficiency. Destroying it would allow the artist to reveal the machine's "flaws" in a way, maybe humanize its outputs, or highlight the subjectivity of what we consider "ideal". A more philosophical question would then be: if ML models are not striving for perfection, are they still fulfilling their purpose? One aspect of this idea was to extract the outputs from intermediate layers of neural networks, and visualize how they would look like if we were to corrupt training data by feeding incomplete, conflicting, or if we were to modify the loss function to not be strictly decreasing.

For the scope of this final project, I specifically explored how to visualize intermediate layers of neural networks, or what techniques there are to extract the filters and feature maps of those mysterious hidden layers that do most of the work, but are almost never exposed. I was interested in understanding how feature maps look as we progress deeper into the network, or how they change together with the loss function. 

## Explorations

There are 3 main aspects of visualizing layers in a neural networks that I explored in this project:  
1. Visualizing a single layer with decreasing loss in a simple 2-layer NN trained on the faces dataset
2. Visualizing feature maps of different layers in a pre-trained model such as VGG and Resnet
3. Amplifying features in differnt layers onto a blank image by using DeepDream

For the first 2 parts, the input image below was used:   
![mosaic_face](https://github.com/user-attachments/assets/63b9a6bb-c83a-4b51-b118-0f4a13021381)

### 1. Single layer with decreasing loss

I used a simple 2 (linear) layered NN, with a ReLU activation function, and visualized the feature maps (or activations) of the first layer of the network during different training epochs. The series of images below from left to right shows the progression of feature maps as the training loss decreases from 39.25 in epoch 0 to 0.29 in epoch 450.

![Intro to ML Final Project-01](https://github.com/user-attachments/assets/1568554f-b320-4026-9d90-610f8746ce00)

Because this is a very simple NN, with a small training sample, we don't expect the feature maps to be very sophisticated or detailed. Yet, you can stil observe that as the training loss decreases, the feature maps become more "focused" in a way where there's less noise in the colored pixels in the last image on the right, compared ot the first image on the left.

### 2. Feature maps of different layers in VGG
Since a simple NN doesn't tend to give detailed feature maps, we try again using a pre-trained VGG16 model. Because VGG16 is a pre-trained model, we are not able to observe the progression of the feature maps as training progresses, but rather we can only visualize the final, optimal feature maps of the trained model. VGG16 is a deep CNN model with 13 convolutional layers (using small 3x3 filters) and 3 fully connected layers, designed for hierarchical feature extraction and image classification. There are 5 main blocks in the model, and the 3 images below represent the output at the end of 3 selected blocks.

**Block 1**
![layer2](https://github.com/user-attachments/assets/95fa51dd-a443-4337-b8cf-15b638d5beaa)

**Block 3**
![layer9](https://github.com/user-attachments/assets/72d3024b-301b-4b89-bbc3-11e0d45d4ee6)

**Block 5**
![layer17](https://github.com/user-attachments/assets/5c72792c-9a1b-4a52-a249-0510a5294a39)

We can see that feature maps closer to the model's input capture fine details of the image (i.e. block 1), while deeper layers progressively abstract these details, resulting in feature maps with less visible detail such as block 5. This behavior is what we'd expect, as the model transforms image features into broader, more generalized concepts for classification. Hence the deeper layers become increasingly hard to interpret (for a human), as we can no longer tell from the output of block 5 that the input image is a human face.

### 3. Amplifying features of different layers on a blank image

Since the large pre-trained models offer much more interesting features, I stuck with using pre-trained models rather than training models myself (for now). Instead of visualizing the learned features of a NN layer in response to an input image, here I tried to amplify certain features in the image by modifying an input image (i.e. using Deep Dream). I wanted the algorithm not be influenced by any existing features in the input image, so I used a completely black and completely white image as the input. Since there are no discernible features in a black or white image, the model should amplify (random) activations in its layers to hopefully reflect the learned features of the model. Basically I was hoping to visualize the types of patterns the model has been trained to recognize or favor, independent of any external input. 

Below are the collated outputs of implementing Deep Dream using specific layers (5, 12, 19, 28, 34, 35) of the VGG19 model, going from left to right, up and down. 

Using a completely black input image:
![Intro to ML Final Project-02](https://github.com/user-attachments/assets/379d1577-8705-4b93-abc7-0a53344a8d67)

Using a completely white input image:
![Intro to ML Final Project-03](https://github.com/user-attachments/assets/eff5347c-d330-4ffc-b752-14743927ed3f)

You can see that the learned features of deeper layers are increasingly more intricate and abstract compared to the shallower layers. The last 2 images (layer 34 and 35) is a comparison between what a ReLU activation (layer 35) does to a Conv2d layer (layer 34). The ReLU layer sort of highlights (or activates) specific parts of the images. 

## Learnings
- **Training Progression**: Feature maps evolve during training epochs, becoming more refined as the model optimizes its weights.
- **Layer Behavior**: Shallow layers tend to focus on fine-grained, more edge-like details, while deeper layers capture more abstract, high-level representations.
- **Large Pre-trained Models**: Pre-trained models, such as VGG or Resnet, offer much more intricate and differentiated feature maps and are hence better suited to derive insights from compared to small custom networks.

