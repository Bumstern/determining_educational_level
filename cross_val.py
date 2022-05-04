import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import KFold, cross_validate
from torchvision import datasets, transforms
from training import *
from fine_tuning import get_model

FINE_TUNED = True


def ft_cross_val(general_dataset: data.Dataset, epochs: int, batch_size: int):
    """ Cross-validation of fine-tuned model

    Args:
        general_dataset: Dataset
        epochs: Amount of epochs
        batch_size: Size of batch

    Returns:
        List of model metrics on each fold: [roc-auc, acc, f1]
    """
    # Divide into folds
    kfold = KFold(n_splits=5, shuffle=True)
    results = []

    # Train and validate ft_model on folds
    for fold, (train_datasplit, test_datasplit) in enumerate(kfold.split(general_dataset)):
        print(f'FOLD #{fold}')
        train_dataloader = data.DataLoader(general_dataset, batch_size, shuffle=False, sampler=train_datasplit)
        test_dataloader = data.DataLoader(general_dataset, 1, shuffle=False, sampler=test_datasplit)

        model = get_model()
        model.to(DEVICE)
        train(model, train_dataloader, test_dataloader, epochs)
        # with open('models/best_model.pth', 'rb') as f:
        #     dictw = torch.load(f)
        #     model.load_state_dict(dictw)
        roc_auc, acc, f1 = test(model, test_dataloader, False)
        print(f'ROC AUC: {roc_auc}, Acc: {acc}, F1: {f1}')
        results.append([roc_auc, acc, f1])

    print('All scores: ', results)
    average = np.average(results, axis=0)
    print(f'K-fold CV average result.\nROC-AUC: {average[0]}, Acc: {average[1]}, F1: {average[2]}')
    return results


def svm_cross_val(X_src: str, y_src: str):
    """ Cross-validation of SVM model

    Args:
        X_src: Path to embeddings
        y_src: Path to labels

    Returns:
        List of model metrics on each fold: [roc-auc, acc, f1]
    """
    X = np.load(X_src)
    y = np.load(y_src)
    print(f'Size of general dataset: {X.shape}')
    classifier = SVC(C=0.3)
    scoring = ['roc_auc', 'accuracy', 'f1']
    result = cross_validate(classifier, X, y, cv=5, scoring=scoring, return_train_score=False)
    return result


def print_cv_res(cv_res_path: str):
    """ Prints average result of cross-validation

    Args:
        cv_res_path: Path to list of model metrics
    """
    ft_cv_res = np.load(cv_res_path)
    print('All scores: ', ft_cv_res)
    average = np.average(ft_cv_res, axis=0)
    print(average.shape)
    print(f'K-fold CV average result.\nROC-AUC: {average[0]}, Acc: {average[1]}, F1: {average[2]}')


def main():
    print(f'Train on {DEVICE}')

    # Cross-validation
    if FINE_TUNED:  # cv of fine-tuned model
        transform = transforms.Compose([transforms.ToTensor()])
        general_dataset = datasets.ImageFolder('../dataset', transform=transform)
        ft_cv_res = ft_cross_val(general_dataset, 30, 16)
        np.save('ft_cv_res_30_epochs.npy', ft_cv_res)
    else:  # cv of SVM
        result = svm_cross_val('embeds_no_norm.npy', 'labels_no_norm.npy')
        print(result)
        roc_auc, acc, f1 = result['test_roc_auc'], result['test_accuracy'], result['test_f1']
        np.save('svm_cv_res.npy', [roc_auc, acc, f1])
        print(
            f'K-fold CV average result.\nROC-AUC: {np.average(roc_auc)}, Acc: {np.average(acc)}, F1: {np.average(f1)}')


if __name__ == "__main__":
    main()
