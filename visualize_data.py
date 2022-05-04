from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt

TSNE_FLAG = False


def main():
    embeds = np.load("embeds/embeds.npy")
    labels = np.load("embeds/labels.npy")
    res_embeds = []

    # Visualize dataset with T-SNE
    if TSNE_FLAG:
        res_embeds = TSNE(n_components=2, learning_rate='auto', perplexity=35).fit_transform(embeds, labels)
        print("T-SNE shape: ", res_embeds.shape)
        np.save('embeds/tsne_embeds.npy', res_embeds)
        # res_embeds = np.load('tsne_embeds.npy')
    else: # with PCA
        res_embeds = PCA(n_components=2).fit_transform(embeds, labels)
        print("PCA shape: ", res_embeds.shape)
        np.save('embeds/pca_embeds.npy', res_embeds)
        # res_embeds = np.load('pca_embeds.npy')

    # Plot a graph
    for point, label in zip(res_embeds, labels):
        x, y = point
        plt.scatter(x, y, c=('red' if label == 1 else 'blue'))
    plt.show()


if __name__=="__main__":
    main()
