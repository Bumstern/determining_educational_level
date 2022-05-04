import itertools
import datetime
import torch
from torch import nn
from torch.utils import data
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn import metrics

# Define the device where we will train on
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'


def save_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer,
                    epoch: int, history: list, best_val_acc: float):
    """ Saves checkpoint to .pth file in /checkpoints directory

    Args:
        model: Neural model
        optimizer: Optimizer (Exmpl: Adam)
        epoch: Epochs amount
        history: List of model metrics
        best_val_acc: Best accuracy on validation set
    """
    dt = datetime.datetime.now()
    dt_string = dt.strftime("%d-%m-%Y %H-%M")
    pathname_pattern = f"checkpoints/{dt_string} {epoch}-epoch.pth"
    with open(pathname_pattern, 'wb') as f:
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch,
            'history': history,
            'best_val_acc': best_val_acc
        }, f)


def load_checkpoint(path: str, model: nn.Module, optimizer: torch.optim.Optimizer) -> tuple[int, list, float]:
    """ Loads checkpoint from .pth file. Configures model and optimizer

    Args:
        path: Path to checkpoint
        model: Neural model (Same that used in saved checkpoint)
        optimizer: Optimizer (Same that used in saved checkpoint)

    Returns:
        List of number of epochs, history, best validation accuracy
    """
    with open(path, 'rb') as f:
        checkpoint = torch.load(f)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch'] + 1
    history = checkpoint['history']
    best_val_acc = checkpoint['best_val_acc']
    return epoch, history, best_val_acc


def epoch_fit(model: torch.nn.Module,
              train_dataloader: data.DataLoader,
              optimizer: torch.optim.Optimizer,
              criterion: torch.nn.CrossEntropyLoss) -> tuple[float, float]:
    """ Training model on 1 epoch

    Args:
        model: Neural model
        train_dataloader: Train dataloader
        optimizer: Optimizer
        criterion: Loss function

    Returns:
        List of epoch loss, epoch accuracy
    """
    # Switch model to train mode
    model.train()

    epoch_loss = 0.0
    epoch_acc = 0.0
    num_of_batches = len(train_dataloader)
    num_of_elems = len(train_dataloader.dataset)

    for inputs, labels in train_dataloader:
        # Send tensors to DEVICE
        inputs = inputs.to(DEVICE)
        labels = labels.to(DEVICE)

        # Reset gradients to zero
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)

        # Backward pass
        loss = criterion(outputs, labels)
        loss.backward()

        # Gradient step
        optimizer.step()

        # Accumulate loss and accuracy
        epoch_loss += loss.item()
        preds = torch.argmax(outputs, 1)
        epoch_acc += sum(preds == labels.data)

    # Calculate average values
    epoch_loss /= num_of_batches
    epoch_acc /= num_of_elems
    return epoch_loss, epoch_acc.cpu()


def epoch_eval(model: nn.Module,
               val_dataloader: data.DataLoader,
               criterion: torch.nn.CrossEntropyLoss) -> tuple[float, float]:
    """ Validate model on 1 epoch

    Args:
        model: Neural model
        val_dataloader: Validation dataloader
        criterion: Loss function

    Returns:
        List of validation loss, validation accuracy
    """
    # Switch model to evaluation mode
    model.eval()

    val_loss = 0.0
    val_acc = 0.0
    num_of_batches = len(val_dataloader)
    num_of_elems = len(val_dataloader.dataset)

    for inputs, labels in val_dataloader:
        # Send tensors to DEVICE
        inputs = inputs.to(DEVICE)
        labels = labels.to(DEVICE)

        # Getting model outputs without calculating gradients
        with torch.no_grad():
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Accumulate validation loss and accuracy
            val_loss += loss.item()
            preds = torch.argmax(outputs, 1)
            val_acc += torch.sum(preds == labels.data)

    # Calculate average values
    val_loss /= num_of_batches
    val_acc /= num_of_elems
    return val_loss, val_acc.cpu()


def train(model: nn.Module,
          train_dataloader: data.DataLoader,
          val_dataloader: data.DataLoader,
          epochs: int,
          checkpoint_path: str = None) -> list:
    """ Train loop

    Args:
        model: Neural model
        train_dataloader: Train dataloader
        val_dataloader: Validation dataloader
        epochs: Number of epochs
        checkpoint_path: Path to checkpoints

    Returns:
        Matrix of metrics and loss on train and val sets for each epoch: [train_loss, train_acc, val_loss, val_acc]
    """
    # Switch model to train mode
    model.train()

    # Define training parameters
    optimizer = torch.optim.Adam(model.parameters(), weight_decay=0.01, amsgrad=True)
    criterion = torch.nn.CrossEntropyLoss()
    history = []
    start_epoch = 0
    best_val_acc = 0.0
    log_template = "\nEpoch {ep:03d} train_loss: {t_loss:0.4f} \
                    val_loss {v_loss:0.4f} train_acc {t_acc:0.4f} val_acc {v_acc:0.4f}"

    # Load checkpoint
    if checkpoint_path is not None:
        start_epoch, history, best_val_acc = load_checkpoint(checkpoint_path, model, optimizer)
        print(f'Checkpoint {checkpoint_path} loaded!')

    # Train loop
    for epoch in tqdm(range(start_epoch, epochs), desc='Epoch', position=0, leave=True):
        # Training
        train_loss, train_acc = epoch_fit(model, train_dataloader, optimizer, criterion)
        # Validation
        val_loss, val_acc = epoch_eval(model, val_dataloader, criterion)
        history.append((train_loss, train_acc, val_loss, val_acc))

        # Create checkpoint every 5 epochs
        if epoch % 5 == 0:
            save_checkpoint(model, optimizer, epoch, history, best_val_acc)

        # Saves best model
        if best_val_acc < val_acc:
            torch.save(model.state_dict(), "models/best_model.pth")
            best_val_acc = val_acc

        tqdm.write(log_template.format(ep=epoch, t_loss=train_loss,
                                       v_loss=val_loss, t_acc=train_acc, v_acc=val_acc))
    return history


def test(model: nn.Module, test_dataloader: data.DataLoader, with_plot: bool = True) -> tuple[float, float, float]:
    """ Testing model

    Args:
        model: Neural model
        test_dataloader: Test dataloader
        with_plot: True - plot a ROC-AUC curve

    Returns:
        List of metrics: [roc_auc, acc, f1]
    """
    # Switch model to evaluating mode
    model.eval()
    y_true = []

    # Test loop
    with torch.no_grad():
        logits = []
        pred_labels = []
        for inputs, labels in tqdm(test_dataloader):
            # Send tensors to DEVICE
            inputs = inputs.to(DEVICE)
            outputs = model(inputs).cpu()

            # Accumulate logits
            logits.append(outputs)
            y_true = list(itertools.chain(y_true, labels))
            pred_labels.append(*torch.argmax(outputs, dim=1).data)

    # Get model probability predictions
    probs = nn.functional.softmax(torch.cat(logits), dim=-1).numpy()
    pred_probs = probs[:, 0]

    # Getting metrics
    fpr, tpr, _ = metrics.roc_curve(y_true, pred_probs, pos_label=0)
    roc_auc = metrics.auc(fpr, tpr)
    acc = metrics.accuracy_score(y_true, pred_labels)
    f1 = metrics.f1_score(y_true, pred_labels, average='binary')

    # Plot ROC-AUC curve
    if with_plot:
        plt.style.use('seaborn')
        plt.title('Receiver Operating Characteristic')
        plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
        plt.legend(loc='lower right')
        plt.plot([0, 1], [0, 1], 'r--')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.show()

    return roc_auc, acc, f1
