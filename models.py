import clip
import torch
from torch import nn


class CLIPModel(nn.Module):
    def __init__(self, model_name="ViT-B/16"):
        super().__init__()
        self.clip, self.preprocess = clip.load(model_name, device="cuda")

    @property
    def dtype(self):
        return self.clip.visual.conv1.weight.dtype

    def encode_image(self, image):
        image_features, _ = self.clip.visual(image.type(self.dtype))
        return image_features

    def encode_text(self, text):
        x = self.clip.token_embedding(text).type(self.dtype)

        x = x + self.clip.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.clip.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.clip.ln_final(x).type(self.dtype)

        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.clip.text_projection

        return x


class EmbeddingModel(nn.Module):
    def __init__(self, input_img_dim, input_txt_dim, hashbit):
        super(EmbeddingModel, self).__init__()
        self.hashbit_image = nn.Sequential(
            nn.Linear(input_img_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, hashbit),
            nn.Tanh()
        )
        self.hashbit_text = nn.Sequential(
            nn.Linear(input_txt_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, hashbit),
            nn.Tanh()
        )
    
    def forward(self, img, text, only_img=False):
        if not only_img:
            hash_img = self.hashbit_image(img)
            hash_text = self.hashbit_text(text)
            return hash_img, hash_text
        else:
            hash_img = self.hashbit_image(img)
            return hash_img
        