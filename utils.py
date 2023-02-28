import torch
from torchvision.transforms import transforms
from PIL import Image
import torchvision.models as models
from torch.nn import MSELoss
from plot_utils import plot_imgs


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_img(img_path, img_size):
    '''
    Returns style and content images as tensors
    '''

    # Load images
    img = Image.open(img_path)

    img_preproc = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(([0.485, 0.456, 0.406]), ([0.229, 0.224, 0.225]))
    ])

    # Preprocess images 
    img = img_preproc(img).unsqueeze(0).to(device)
    return img


def extract_feats(img, model, layers: set):

    feat_img = img

    features = {}
    model_modules = list(model.modules())[0]

    for layer_no, layer in enumerate(model_modules):
        feat_img = layer(feat_img)

        if layer_no in layers:
            features[layer_no] = feat_img

    return features


def gram_matrix(activations):
    '''
    Construct gram matrix consisting of inner product of ith feature map 
    with jth feature map. 
    '''
    _, num_channels, height, width = activations.shape
    activations = activations.view(num_channels, height*width)
    
    return torch.mm(activations, activations.t())


def style_loss(style_gram, input_features):

    G_style = style_gram
    G_input = gram_matrix(input_features)
    loss = MSELoss()
    return loss(G_style, G_input)


def content_loss(content_features, input_features):
    loss = MSELoss()
    return loss(content_features, input_features)


