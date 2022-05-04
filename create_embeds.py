from torchvision import datasets, transforms
import numpy as np
import torch.nn as nn
import torch.utils.data as data
import torch
from tqdm import tqdm

from lib import VGGFace


def get_model():
    """ Get feature-extracting model

    Returns:
        Feature-extracting model
    """
    # Build VGGFace model and load pre-trained weights
    model = VGGFace()
    model_dict = torch.load('models/vggface.pth', map_location=lambda storage, loc: storage)
    model.load_state_dict(model_dict)
    model.eval()

    # Fine-tuning
    model.fc['fc7-relu'] = nn.Identity()
    model.fc['fc8'] = nn.Identity()

    # Freezing weights
    for param in model.parameters():
        param.requires_grad = False

    return model


def create_embeddings(model, dataloader: data.DataLoader, path_embed: str, path_labels: str):
    """ Creates embeddings and labels to files

    Args:
        model: Neural model
        dataloader: Dataloader
        path_embed: Path to embeddings to save
        path_labels: Path to labels to save
    """
    embeddings = []
    labels = []
    for batch in tqdm(dataloader):
        data, label = batch
        embeddings.append(*model(data).tolist())
        labels.append(*label.tolist())
    np.save(path_embed, embeddings)
    np.save(path_labels, labels)


def main():
    model = get_model()
    print(model)
    print('Getting dataloaders...')
    dataset = datasets.ImageFolder('../dataset', transform=transforms.Compose([transforms.ToTensor()]))
    dataloader = data.DataLoader(dataset, batch_size=1, shuffle=False)
    print('Done!')

    print('Create embeddings...')
    model.eval()
    create_embeddings(model, dataloader, 'embeds/embeds.npy', 'embeds/labels.npy')
    print('Done!')

    embed = np.load('test_embeds_no_norm.npy')
    print('Test info:')
    print(embed.shape)


if __name__ == "__main__":
    main()
