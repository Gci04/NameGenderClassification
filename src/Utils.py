import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split


def get_data(language="en"):
    """Loads data from .csv files and split to train & test set """

    if language == "ru" :
        df = pd.read_csv('../Data/Russian/data.csv.zip', compression='zip',usecols=[1,3] ,names=["имя","пол"], sep=',')

        df.dropna(axis=0,inplace=True)
        df.drop_duplicates(inplace=True)
        df.query('пол == "Ж" | пол == "М"',inplace=True)

        train , test = train_test_split(df, test_size= 0.2,shuffle=True)

        train.reset_index(drop=True,inplace=True)
        test.reset_index(drop=True,inplace=True)

    else :
        train_data = pd.read_csv("../Data/English/train_eng.csv")
        test_data = pd.read_csv("../Data/English/test_eng.csv")

        # Convert data to numpy arrays
        train = train_data.values
        test = test_data.values

        train = np.stack(sorted(list(train), key=lambda x: len(x[0])))

    return train, test

def transform_data(data, max_len):
    """
    Transform the data into machine readable format.
    Parameters:
    ----------
        data: ndarray
            first column is name, and the second is gender
        max_len: int
            maximum length of a name
    Return :
        names : ndarray with shape [?,max_len]
        labels : ndarray with shape [?,1]
        vocab : dictionary with mapping from letters to integer IDs
    """

    unique = list(set("".join(data[:,0])))
    unique.sort()
    vocab = dict(zip(unique, range(1,len(unique)+1))) # start from 1 for zero padding

    classes = list(set(data[:,1]))
    classes.sort()
    class_map = dict(zip(classes, range(len(unique))))

    names = list(data[:,0])
    labels = list(data[:,1])

    def transform_name(name):
        point = np.zeros((1, max_len), dtype=int)
        name_mapped = np.array(list(map(lambda l: vocab[l], name)))
        point[0,0: len(name_mapped)] = name_mapped
        return point

    transform_label = lambda lbl: np.array([[class_map[lbl]]])

    names = list(map(transform_name, names))
    labels = list(map(transform_label, labels))

    names = np.concatenate(names, axis=0)
    labels = np.concatenate(labels, axis=0)

    return names, labels, vocab

def get_minibatches(names, labels, mb_size):
    """
    Split data in minibatches
    Parameters:
        names: ndarray of shape [?, max_name_len]
        labels: ndarray of shape [?, 1]
        mb_size: batch size

    Return:
        batches : list
            list of minibatches
    """
    batches = []

    position = 0

    while position + mb_size < len(labels):
        batches.append((names[position: position + mb_size], labels[position: position + mb_size]))
        position += mb_size

    batches.append((names[position:], labels[position:]))

    return batches

def plot_losses(title,pl_acc,pl_loss,epochs):
    N = np.arange(0, epochs)
    plt.style.use('ggplot')
    plt.figure(1,figsize=(12,8))

    plt.subplot(211)
    plt.title(title)
    plt.plot(N, pl_acc[:,0],label="Train Accuracy", linewidth=2)
    plt.plot(N, pl_loss[:,0],label="Train Loss",linewidth=2,)
    plt.xlim(-5,55)
    plt.legend(loc=5)
    plt.ylabel("Loss/Accuracy")

    plt.subplot(212)
    plt.plot(N, pl_loss[:,1],label="Test Loss",linewidth=2)
    plt.plot(N, pl_acc[:,1],label="Test Accuracy",linewidth=2)
    plt.xlabel("Epoch #")
    plt.xlim(-5,55)
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc=5)

    plt.show()
    # plt.savefig("tes.png")
