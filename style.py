import torch
import torchvision
import torchvision.models as models
from tqdm import tqdm
from utils import load_img, extract_feats, gram_matrix, content_loss, style_loss
import argparse
from plot_utils import plot_imgs
import os
from datetime import datetime

cuda_avail = torch.cuda.is_available()
device = torch.device("cuda" if cuda_avail else "cpu")


def main(args):
    aspect_ratio = args.a_ratio
    if aspect_ratio > 1:
        w = 512
        h = int(w/aspect_ratio)
    else:
        h = 512
        w = int(h*aspect_ratio)
    img_size = (h, w)
    
    style_img_path = args.s_path
    content_img_path = args.c_path
    
    content = load_img(content_img_path, img_size)
    style = load_img(style_img_path, img_size) 
    input = content.clone().to(device)
    
    # Load pretrained model
    vgg19 = models.vgg19(pretrained=True).features.eval()
    vgg19.to(device)
    
    # Initialize style weights: relative weight for each layer
    style_weights = {0: 1, 5: 0.75, 10: 0.2, 19: 0.2, 28: 0.2}

    # Initialize which layers will be used for styling, and which for content
    style_layers = set({0, 5, 10, 19, 28})
    content_layers = set({21})
    input_layers = set({0, 5, 10, 19, 21, 28})

    # Freeze model parameters 
    input.requires_grad_(True)
    for param in vgg19.parameters():
        param.requires_grad_(False)

    # Extract features (activation maps) from chosen layers
    content_feats = extract_feats(content, vgg19, content_layers)
    style_feats = extract_feats(style, vgg19, style_layers)

    # Produce gram matrix using style features
    S_grams = {}
    for style_layer in style_layers:
        S_grams[style_layer] = gram_matrix(style_feats[style_layer])

    # Initialize hyperparameters
    if args.epochs:
        num_epochs = args.epochs
    else:
        num_epochs = 3000

    optimizer = torch.optim.Adam([input])
    alpha = args.alpha
    beta = args.beta

    # Create folder to store results
    now = datetime.now()
    date_str = now.strftime("%d-%m-%Y_%H-%M-%S")
    out_path = "Results\\" + date_str
    os.makedirs(out_path)

    # Save hyperparameters 
    with open(out_path + '/config.txt', 'w') as f:
        l_eps = "Epochs: " + str(num_epochs) + '\n'
        l_alpha = "alpha: " + str(alpha) + '\n'
        l_beta = "beta: " + str(beta) + '\n'
        l_style_img = "Style image path: " + style_img_path + '\n'
        l_content_img = "Content image path: " + content_img_path 
        f.writelines([l_eps, l_alpha, l_beta, l_style_img, l_content_img])

    # Main training loop
    for epoch in tqdm(range(num_epochs), desc="Progress..."):
        input_feats = extract_feats(input, vgg19, input_layers)

        # Compute 
        cum_content_loss = 0
        cum_style_loss = 0

        for content_layer in content_layers:
            cum_content_loss += content_loss(content_feats[content_layer], input_feats[content_layer])

        for style_layer in style_layers:
            layer_weight = style_weights[style_layer]
            cum_style_loss += layer_weight * style_loss(S_grams[style_layer], input_feats[style_layer])

        # Normalize style loss as in paper
        _, num_channels, height, width = input_feats[style_layer].shape
        cum_style_loss /= num_channels * height * width

        # Compute total weighted loss
        total_loss = alpha*cum_content_loss + beta*cum_style_loss

        # Backprop and zero gradients
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        if  (epoch + 1) % 50 == 0:
            print('\nTotal loss: ', total_loss.item())
            plot_imgs(input, seq_no=(epoch + 1) // 50, lp_path=out_path, log_process=args.lp)
    
    # Save result
    img = torchvision.transforms.ToPILImage()(img)
    img.save('{folder}\\{file}.jpg'.format(folder=out_path, file=args.o_filename))   

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='Neural Style Transfer', description='Combine content and style of two images.')
    parser.add_argument('--s_path', type=str, required=True, help='Style image filepath.')
    parser.add_argument('--c_path', type=str, required=True, help='Content image filepath.')
    parser.add_argument('--o_filename', type=str, required=True, help='Output image filename.')
    parser.add_argument('--a_ratio', type=float, required=True, help='Desired aspect ratio for output image (w/h).')
    parser.add_argument('--epochs', type=int, help='Optional: Number of epochs to perform style transfer for (default: 4000 steps).')
    parser.add_argument('--alpha', type=float, required=True, help='Weight for content loss.')
    parser.add_argument('--beta', type=float, required=True, help='Weight for style loss.')
    parser.add_argument('--lp', type=bool, required=True, help='Save a sequence of images that show the style transfer process.')

    args = parser.parse_args()
    main(args)