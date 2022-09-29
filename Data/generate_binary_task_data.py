'''
To prepare binarized task data, given two labels, e.g. 7 and 8, extract all training/testing data
with label 7 and 8; keep the labels of a fixed number of training data (~1% of total)
'''

import os
import numpy as np
import pandas as pd
from torchvision import datasets
from torchvision.transforms import ToTensor
import tensorflow_datasets as tfds
from PIL import Image

DATA_ROOT = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'raw_data')
TASK_ROOT = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'task_data')

if not os.path.exists(DATA_ROOT):
    os.mkdir(DATA_ROOT)

if not os.path.exists(TASK_ROOT):
    os.mkdir(TASK_ROOT)


# Save task data to specific directory
def save_task_data(parent_dir, X_l, y_l, X_u, y_u, X_test, y_test, task_dir):
    if not os.path.exists(parent_dir):
        os.mkdir(parent_dir)
    task_dir = os.path.join(parent_dir, task_dir)
    if not os.path.exists(task_dir):
        os.mkdir(task_dir)
    np.save(os.path.join(task_dir, 'X_l.npy'), X_l)
    np.save(os.path.join(task_dir, 'y_l.npy'), y_l)
    np.save(os.path.join(task_dir, 'X_u.npy'), X_u)
    np.save(os.path.join(task_dir, 'y_u.npy'), y_u)
    np.save(os.path.join(task_dir, 'X_test.npy'), X_test)
    np.save(os.path.join(task_dir, 'y_test.npy'), y_test)
    return


# Extract data for a specific binary classification task for MNIST and Fashion
def split_for_binary_task(train_data, test_data, c0=0, c1=1, num_labeled=50):
    # Prepare bianrized training data; within the training data, a given number of data is labeled
    train_data_bin_0 = []
    train_data_bin_1 = []
    for i in range(len(train_data)):
        if train_data[i][1] == c0:
            train_data_bin_0.append([train_data[i][0].numpy(), 0])
        elif train_data[i][1] == c1:
            train_data_bin_1.append([train_data[i][0].numpy(), 1])
    train_data_bin_labeled = train_data_bin_0[0: num_labeled // 2] + train_data_bin_1[0: num_labeled // 2]
    train_data_bin_unlabeled = train_data_bin_0[num_labeled // 2:] + train_data_bin_1[num_labeled // 2:]
    X_l = np.array([x[0] for x in train_data_bin_labeled])
    y_l = np.array([x[1] for x in train_data_bin_labeled])
    X_u = np.array([x[0] for x in train_data_bin_unlabeled])
    y_u = np.array([x[1] for x in train_data_bin_unlabeled])

    # Prepare binarized testing data
    test_data_bin = []
    for i in range(len(test_data)):
        if test_data[i][1] == c0:
            test_data_bin.append([test_data[i][0].numpy(), 0])
        elif test_data[i][1] == c1:
            test_data_bin.append([test_data[i][0].numpy(), 1])
    X_test = np.array([x[0] for x in test_data_bin])
    y_test = np.array([x[1] for x in test_data_bin])

    return X_l, y_l, X_u, y_u, X_test, y_test


# Generate task data for binarized MNIST, Fashion and CIFAR10
# Each task is a binary classification on two classes, 45 tasks in total for each dataset
def generate_binary_task_data(dataset='mnist'):

    if dataset == 'mnist':
        task_parent_dir = os.path.join(TASK_ROOT, 'mnist_bin')
        train_data = datasets.MNIST(root=DATA_ROOT, train=True, download=True, transform=ToTensor())
        test_data = datasets.MNIST(root=DATA_ROOT, train=False, download=True, transform=ToTensor())
        num_labeled_data = 120
    elif dataset == 'fashion':
        task_parent_dir = os.path.join(TASK_ROOT, 'fashion_bin')
        train_data = datasets.FashionMNIST(root=DATA_ROOT, train=True, download=True, transform=ToTensor())
        test_data = datasets.FashionMNIST(root=DATA_ROOT, train=False, download=True, transform=ToTensor())
        num_labeled_data = 120
    elif dataset == 'cifar10':
        task_parent_dir = os.path.join(TASK_ROOT, 'cifar10_bin')
        train_data = datasets.CIFAR10(root=DATA_ROOT, train=True, download=True, transform=ToTensor())
        test_data = datasets.CIFAR10(root=DATA_ROOT, train=False, download=True, transform=ToTensor())
        num_labeled_data = 400
    else:
        raise NotImplementedError

    if not os.path.exists(task_parent_dir):
        os.mkdir(task_parent_dir)

    # Take two different classes of data to form a distinct binary classification task
    # No swapping of labels allowed, i.e. if there is a task with classes 4 and 5, no future task can be 5 and 4
    for c0 in range(9):
        for c1 in range(c0 + 1, 10):
            X_l, y_l, X_u, y_u, X_test, y_test = split_for_binary_task(train_data, test_data, c0=c0, c1=c1, num_labeled=num_labeled_data)
            save_task_data(task_parent_dir, X_l, y_l, X_u, y_u, X_test, y_test, str(c0) + '_' + str(c1))
            print(dataset + ': task saved for ' + str(c0) + ' vs ' + str(c1))

    return


# Prepare data for one multiclass classification of CIFAR100
# e.g. if original label == 11, it will belong to task_num == 1, new label == 1
def split_for_cifar100_task(train_data, test_data, task_num=0, num_labeled_per_class=5):
    train_data_split_classes = [[] for _ in range(10)]
    for i in range(len(train_data)):
        if task_num * 10 <= train_data[i][1] <= task_num * 10 + 9:
            new_label = train_data[i][1] - task_num * 10
            train_data_split_classes[new_label].append([train_data[i][0].numpy(), new_label])
    # split X_l and X_u for the task
    train_data_split_labeled = []
    train_data_split_unlabeled = []
    for x in train_data_split_classes:
        train_data_split_labeled += x[0: num_labeled_per_class]
        train_data_split_unlabeled += x[num_labeled_per_class:]

    X_l = np.array([x[0] for x in train_data_split_labeled])
    y_l = np.array([x[1] for x in train_data_split_labeled])
    X_u = np.array([x[0] for x in train_data_split_unlabeled])
    y_u = np.array([x[1] for x in train_data_split_unlabeled])

    test_data_split = []
    for i in range(len(test_data)):
        if task_num * 10 <= test_data[i][1] <= task_num * 10 + 9:
            new_label = test_data[i][1] - task_num * 10
            test_data_split.append([test_data[i][0].numpy(), new_label])

    X_test = np.array([x[0] for x in test_data_split])
    y_test = np.array([x[1] for x in test_data_split])

    return X_l, y_l, X_u, y_u, X_test, y_test


# Generate task data for multiclass classification on CIFAR100, with 10 classes a task
# 100 classes (10 tasks) in total, each class has 500 training and 100 testing
# Each task has: 5000 training data, within the 5000, 200 labeled, 4800 unlabeled
def generate_cifar100_data():
    task_parent_dir = os.path.join(TASK_ROOT, 'cifar100')
    train_data = datasets.CIFAR100(root=DATA_ROOT, train=True, download=True, transform=ToTensor())
    test_data = datasets.CIFAR100(root=DATA_ROOT, train=False, download=True, transform=ToTensor())

    for task_num in range(10):
        X_l, y_l, X_u, y_u, X_test, y_test = split_for_cifar100_task(train_data, test_data, task_num=task_num,
                                                                     num_labeled_per_class=200)
        save_task_data(task_parent_dir, X_l, y_l, X_u, y_u, X_test, y_test, str(task_num))
        print('CIFAR100: task saved for ' + str(task_num))

    return


# helper function of omniglot data
# create pandas dataframe of omniglot
# cols = 'alphabet', 'alphabet_char_id', 'image', 'label'
# min of alphabet = 0, max = 48
# each image is a numpy array of shape (105, 105, 3)
# need convert rgb to grayscale and then np.moveaxis(img, -1, 0) for (N, C, H, W)
def generate_omniglot_data_helper(df: pd.DataFrame):
    for alphabet_id in range(0, 50):
        print("Generating data for alphabet " + str(alphabet_id))
        alphabet_train_df = df.loc[df['alphabet'] == alphabet_id]
        alphabet_train_df = alphabet_train_df.sort_values(['alphabet_char_id'])
        images = []
        labels = []
        for index, row in alphabet_train_df.iterrows():
            image = row['image']
            image = np.dot(image[..., :3], [0.299, 0.587, 0.114])
            image = np.moveaxis(image, -1, 0)
            images.append(image)
            labels.append(row['alphabet_char_id'])
        images = np.array(images)
        labels = np.array(labels)
        task_dir = os.path.join(TASK_ROOT, 'omniglot', str(alphabet_id))
        if not os.path.exists(task_dir):
            os.mkdir(task_dir)
        np.save(os.path.join(task_dir, 'X.npy'), images)
        np.save(os.path.join(task_dir, 'y.npy'), labels)
    return


# Generate task data for Omniglot; do not split X_l, X_u for now
# Each alphabet has 14 to 55 distinct characters
# Do 10-way-N_l-shot classification, i.e. classification on first 10 distinct chars (200 data)
# with N_l training examples per class, N_l: N_u: N_test = 5: 11: 4
def generate_omniglot_data():
    parent_dir = os.path.join(TASK_ROOT, 'omniglot')
    if not os.path.exists(parent_dir):
        os.mkdir(parent_dir)
    ds = tfds.load('omniglot', split=None, shuffle_files=False, download=True, data_dir=DATA_ROOT)
    omniglot_df_background = pd.DataFrame(tfds.as_dataframe(ds['train']))
    omniglot_df_eval = pd.DataFrame(tfds.as_dataframe(ds['test']))
    omniglot_df_all = pd.concat([omniglot_df_background, omniglot_df_eval], ignore_index=True)
    generate_omniglot_data_helper(omniglot_df_all)
    return


def split_for_omniglot_task(n_l=5, n_u=11):
    parent_dir = os.path.join(TASK_ROOT, 'omniglot')
    for alphabet in range(50):
        print("Splitting Omniglot data for alphabet " + str(alphabet))
        task_dir = os.path.join(parent_dir, str(alphabet))
        X = np.load(os.path.join(task_dir, 'X.npy'))
        y = np.load(os.path.join(task_dir, 'y.npy'))
        X_split = []
        y_split = []
        for label in range(10):
            X_split.append(X[y == label])
            y_split.append(y[y == label])
        X_l = [x[0: n_l] for x in X_split]
        y_l = [x[0: n_l] for x in y_split]
        X_u = [x[n_l: n_l + n_u] for x in X_split]
        y_u = [x[n_l: n_l + n_u] for x in y_split]
        X_test = [x[n_l + n_u:] for x in X_split]
        y_test = [x[n_l + n_u:] for x in y_split]
        X_l = np.expand_dims(np.concatenate(X_l, axis=0), axis=1)
        y_l = np.concatenate(y_l)
        X_u = np.expand_dims(np.concatenate(X_u, axis=0), axis=1)
        y_u = np.concatenate(y_u)
        X_test = np.expand_dims(np.concatenate(X_test, axis=0), axis=1)
        y_test = np.concatenate(y_test)
        save_task_data(parent_dir=parent_dir, X_l=X_l, y_l=y_l, X_u=X_u, y_u=y_u,
                       X_test=X_test, y_test=y_test, task_dir=str(alphabet))
    return


# Input a 4D array (N, C, H, W) and resize each image, so far only support C == 1
def resize_image(X: np.ndarray, target_size=(28, 28)):
    X_resized = []
    for i in range(X.shape[0]):
        x = np.squeeze(X[i])
        img = Image.fromarray(x)
        img = img.resize(size=target_size)
        x_resized = np.array(img)
        X_resized.append(np.expand_dims(x_resized, axis=0))
    return np.array(X_resized)


def resize_omniglot_data(target_size=(28, 28)):
    parent_dir = os.path.join(TASK_ROOT, 'omniglot')
    for alphabet in range(50):
        print("Resizing Omniglot data for alphabet " + str(alphabet))
        task_dir = os.path.join(parent_dir, str(alphabet))
        X_l = np.load(os.path.join(task_dir, 'X_l.npy'))
        X_u = np.load(os.path.join(task_dir, 'X_u.npy'))
        X_test = np.load(os.path.join(task_dir, 'X_test.npy'))
        X_l_resized = resize_image(X_l, target_size)
        X_u_resized = resize_image(X_u, target_size)
        X_test_resized = resize_image(X_test, target_size)
        np.save(os.path.join(task_dir, 'X_l_small.npy'), X_l_resized)
        np.save(os.path.join(task_dir, 'X_u_small.npy'), X_u_resized)
        np.save(os.path.join(task_dir, 'X_test_small.npy'), X_test_resized)
    return


if __name__ == '__main__':
    for dataset in ['mnist', 'cifar10']:
        generate_binary_task_data(dataset)