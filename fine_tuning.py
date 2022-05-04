from math import ceil, floor
from torchvision import datasets, transforms
import numpy as np
from lib import VGGFace
from training import *


def get_model():
    """ Get fine-tuned model

    Returns:
        Pretrained fine-tuned model
    """
    # Build VGGFace model and load pre-trained weights
    model = VGGFace()
    model_dict = torch.load('models/vggface.pth', map_location=lambda storage, loc: storage)
    model.load_state_dict(model_dict)
    model.eval()

    # Freezing weights
    for param in model.parameters():
        param.requires_grad = False

    # Fine-tuning
    model.fc['fc8'] = nn.Linear(in_features=4096, out_features=2)
    return model


def get_datasets(path: str, transform: transforms = None, with_val: bool = False, seed: int = None):
    """ Get train, test and val dataset

        Args:
            path: Path to dataset
            transform: Dataset transformation (augmentation)
            with_val: Divide dataset into train, test, val with "True" value.
                      Otherwise return only train and test
            seed: Seed for splitting

        Returns:
            Divided dataset into train, test, (val)
    """
    general_dataset = datasets.ImageFolder(path, transform=transform)
    dataset_size = len(general_dataset)

    if seed is not None:
        torch.manual_seed(seed)

    if with_val: # Train, test, val
        train_dataset, test_val_dataset = data.random_split(general_dataset,
                                                        (ceil(dataset_size * 0.6), floor(dataset_size * 0.4)))
        test_val_size = len(test_val_dataset)
        test_dataset, val_dataset = data.random_split(test_val_dataset, (ceil(test_val_size * 0.5),
                                                                        floor(test_val_size * 0.5)))
        return train_dataset, test_dataset, val_dataset
    else:   # Train, test
        train_dataset, test_dataset = data.random_split(general_dataset,
                                                        (ceil(dataset_size * 0.8), floor(dataset_size * 0.2)))
        return train_dataset, test_dataset


def history_plot(history: list):
    """ Plots a history graph

    Args:
        history: List of values consisted of loss, accuracy
                 on train and val samples
    """
    loss, acc, val_loss, val_acc = zip(*history)

    plt.style.use('seaborn')
    fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, sharex=True)

    ax1.plot(loss, label="train loss")
    ax1.plot(val_loss, label="val loss")
    ax1.legend()
    ax1.set_ylabel('Loss')

    ax2.plot(acc, label='train accuracy')
    ax2.plot(val_acc, label='val accuracy')
    ax2.legend()
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')

    plt.tight_layout()
    plt.show()


def main():
    print(f'Train on {DEVICE}')
    batch_size = 20
    print(f'Batch size: {batch_size}')

    # Getting train, test, val datasets
    train_dataset, test_dataset, val_dataset = get_datasets('../dataset',
                                                            transform=transforms.Compose([transforms.ToTensor()]),
                                                            with_val=True,
                                                            seed=72)
    train_dataloader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = data.DataLoader(test_dataset, shuffle=False)
    val_dataloader = data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Getting model
    model = get_model()
    print(model, '\n')
    print(f'All layers: {len(model.state_dict())}')
    print('Activated layers:')
    for i, layer in enumerate(model.parameters(), start=1):
        if layer.requires_grad:
            print(i, layer.requires_grad)

    # Train model
    model.train()
    model.to(DEVICE)
    history = train(model, train_dataloader, val_dataloader, 50, 16, checkpoint_path=None)
    np.save('history.npy', history) # Save loss and accuracy

    # Plotting history
    history_plot(history)

    # Load weights of best fine-tuned model
    # with open('models/best_model 50 epochs.pth', 'rb') as f:
    #     dictw = torch.load(f)
    #     model.load_state_dict(dictw)

    # Calculate metrics
    print('\nCalculate metrics:')
    model.to('cpu')
    roc_auc = test(model, test_dataloader)
    print(f'Roc-auc: {roc_auc}')


if __name__ == "__main__":
    main()
