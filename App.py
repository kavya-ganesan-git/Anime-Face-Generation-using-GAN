import streamlit as st
import torch
import torch.nn.functional as F
from torchvision.utils import make_grid
from torchvision.transforms import ToPILImage


class Generator(torch.nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(128, 512, 4, 1, 0, bias=False),
            torch.nn.BatchNorm2d(512),
            torch.nn.ReLU(True),
            torch.nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(True),
            torch.nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(True),
            torch.nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(True),
            torch.nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),
            torch.nn.Tanh()
        )

    def forward(self, input):
        return self.main(input)

generator = Generator()
state_dict = torch.load('C:\\Users\\Goodday\\Downloads\\G_.pth', map_location=torch.device('cpu'))


new_state_dict = {}
for k, v in state_dict.items():
    new_key = 'main.' + k if not k.startswith('main.') else k
    new_state_dict[new_key] = v


generator.load_state_dict(new_state_dict)
generator.eval()


def generate_image():
    noise = torch.randn(1, 128, 1, 1)
    with torch.no_grad():
        fake_image = generator(noise).detach().cpu()
    return fake_image

st.title("DCGAN Image Generator")

if st.button("Generate Image"):
 
    fake_image = generate_image()
    image = ToPILImage()(fake_image.squeeze())

    st.image(image, caption="Generated Image", use_column_width=True)
