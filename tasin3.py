Python 3.11.9 (tags/v3.11.9:de54cf5, Apr  2 2024, 10:12:12) [MSC v.1938 64 bit (AMD64)] on win32
Type "help", "copyright", "credits" or "license()" for more information.
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import copy

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load and preprocess image
def load_image(img_path, max_size=400, shape=None):
    image = Image.open(img_path).convert('RGB')

    if max(image.size) > max_size:
        size = max_size
    else:
        size = max(image.size)

    if shape is not None:
        transform = transforms.Compose([
            transforms.Resize(shape),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize(size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    image = transform(image).unsqueeze(0)
    return image.to(device)

# De-normalize and show image
def imshow(tensor, title=None):
    image = tensor.cpu().clone().squeeze(0)
    image = image * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1) + \
            torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    image = image.clamp(0, 1)
    plt.imshow(image.permute(1, 2, 0))
    if title:
        plt.title(title)
    plt.pause(0.001)

# Loss classes
class ContentLoss(nn.Module):
    def _init_(self, target):
        super(ContentLoss, self)._init_()
        self.target = target.detach()

    def forward(self, x):
        self.loss = nn.functional.mse_loss(x, self.target)
        return x

def gram_matrix(input):
    b, c, h, w = input.size()
    features = input.view(c, h * w)
    G = torch.mm(features, features.t())
    return G.div(c * h * w)

class StyleLoss(nn.Module):
    def _init_(self, target_feature):
        super(StyleLoss, self)._init_()
        self.target = gram_matrix(target_feature).detach()

    def forward(self, x):
        G = gram_matrix(x)
        self.loss = nn.functional.mse_loss(G, self.target)
        return x

# Load content and style images
content_img = load_image("content.jpeg")
style_img = load_image("style.jpg", shape=content_img.shape[-2:])

# Display images
plt.figure(); imshow(content_img, title='Content Image')
plt.figure(); imshow(style_img, title='Style Image')

# Load model
cnn = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features.to(device).eval()

# Layers to use
content_layers = ['conv_4']
style_layers = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

# Build model with loss
def get_style_model_and_losses(cnn, style_img, content_img):
    cnn = copy.deepcopy(cnn)
    model = nn.Sequential()
    content_losses, style_losses = [], []

    i = 0
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = f'conv_{i}'
        elif isinstance(layer, nn.ReLU):
            name = f'relu_{i}'
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = f'pool_{i}'
        elif isinstance(layer, nn.BatchNorm2d):
            name = f'bn_{i}'
        else:
            continue

        model.add_module(name, layer)

        if name in content_layers:
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module(f"content_loss_{i}", content_loss)
            content_losses.append(content_loss)

        if name in style_layers:
            target_feature = model(style_img).detach()
            style_loss = StyleLoss(target_feature)
            model.add_module(f"style_loss_{i}", style_loss)
            style_losses.append(style_loss)
... 
...     return model, style_losses, content_losses
... 
... # Run style transfer
... input_img = content_img.clone()
... model, style_losses, content_losses = get_style_model_and_losses(cnn, style_img, content_img)
... optimizer = optim.LBFGS([input_img.requires_grad_()])
... 
... print("🎨 Running style transfer...")
... num_steps = 300
... 
... for step in range(num_steps):
...     def closure():
...         optimizer.zero_grad()
...         model(input_img)
...         style_score = sum(sl.loss for sl in style_losses)
...         content_score = sum(cl.loss for cl in content_losses)
...         loss = style_score + content_score
...         loss.backward()
...         return loss
... 
...     optimizer.step(closure)
...     if step % 50 == 0:
...         print(f"Step {step}/{num_steps}")
... 
... # Save and display
... plt.figure()
... imshow(input_img, title='Stylized Image')
... 
... output = input_img.clone().squeeze(0).cpu()
... output = output * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1) + \
...          torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
... output = output.clamp(0, 1)
... output_img = transforms.ToPILImage()(output)
... output_img.save("output_stylized.jpg")
... 
... print("✅ Output saved as: output_stylized.jpg")
... plt.ioff()
