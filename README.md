# Neural Style Transfer 

This project is an implementation of the paper titled [A Neural Algorithm of Artistic Styles](https://arxiv.org/abs/1508.06576), by Leon A. Gatys, Alexander S. Ecker, and Matthias Bethge. Style transfer uses representations learned by a pre-trainined network to combine the style of one image with the content of another. 

Different layers of a deep neural network learn different features of the input image. For instance, the activation maps produced by early convolutional layers generally correspond to lower level features, while deeper layers learn high level <i>content</i> of the image.

In style transfer, we first pass an image with the desired style through the neural network. We can select the appropriate layers in a network and incorporate their activations into an objective function. We do the same for an image with the desired content, typically selecting a different set of layers (usually a single layer deep in the network is sufficient) to obtain the network's learned representation of the content. Again, this representation is incoroprated into some objective function.

We subsequently train an <i>input</i> image, rather than neural network weights, by enforcing that its content representation should be similar to that of the content image, and its style to that of the style image (we use a gram matrix for this - more details can be found in the paper). 

The final result is an image containing the desired content, but reconstructed in the style of the style image.

## Demo 
![](https://github.com/ZakariaPZ/Style-Transfer/blob/main/CN.gif)

## Usage
To use this program:
- Clone this repository 
- Run the following in your terminal:

```
python .\style.py --s_path 'style image path' --c_path  'content image path' --o_filename 'image name' --epochs 'num epochs' --alpha 'alpha value' --beta 'beta value' --lp 'boolean'
```

