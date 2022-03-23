import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold


#### CONSTANT DECLARATION

NUM_OBSERVATIONS = 500
MAX_ITERS = 20            
LOSS_THRESHOLD = 1e-1
LEARNING_RATE = 1e-3
TEST_RATIO = 0.3
SEED = 666            # seed for spilit dataset
N_SPLITS = 5          # kfold hyperparameter
PLT_DELTA = 2000      # plot delta

def gen_data(num_observations: int = 500):
    """Generate data from multivariate gaussian distribution.
    
    Args:
        num_observations (int): num_observations
    """
    np.random.seed(12)
    x1 = np.random.multivariate_normal([0, 0], [[1, .75], [.75, 1]], num_observations)
    x2 = np.random.multivariate_normal([1, 4], [[1, .75], [.75, 1]], num_observations)

    X = np.vstack((x1, x2)).astype(np.float32)
    Y = np.hstack((np.zeros(num_observations), np.ones(num_observations)))
    return X, Y

def plot(X: np.ndarray, Y: np.ndarray, w: np.ndarray):
    """Plot the data and the decision boundary.

    Args:
        X (array): (n, 2)      data points
        Y (array): (n, 1)      label for x
        w (array): (w0,w1,b)   parameters of the model
    """
    global NUM_OBSERVATIONS
    fig, ax = plt.subplots()
    ax.scatter(X[:, 0], X[:, 1], c=Y, alpha=0.5)
    xmin, xmax = X[:, 0].min() - 1, X[:, 0].max() + 1
    ymin, ymax = X[:, 1].min() - 1, X[:, 1].max() + 1
    ax.set(
        xlabel="X1",
        ylabel="X2",
        title="Basic Linear Classifier",
        xlim=(xmin, xmax),
        ylim=(ymin, ymax),
    )
    xx = np.linspace(xmin, xmax, NUM_OBSERVATIONS)
    yy = (0.5 - w[2] - xx * w[0]) / w[1]
    
    ax.text(
        xmin + 1,
        ymax - 1,
        f"acc = {np.sum((w[0:2].dot(X[:, 0:2].T) + w[-1] > 0.5) == Y.astype(bool)) / len(X)}",
        c="r",
    )

    ax.plot(xx, yy, 'r-')
    plt.show()

def plot_confusion_matrix(cm: np.ndarray):
    """Plot the confusion matrix."""
    fig = plt.figure()
    ax = sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    ax.set(
        title="Confusion Matrix for Perceptron",
        xlabel="Predicted Class",
        ylabel="True Class",
    )
    plt.show()

### Generate data

X, Y = gen_data(NUM_OBSERVATIONS)
np.random.seed(SEED)
test_idx = np.random.choice(len(Y), size=int(len(Y) * TEST_RATIO), replace=False)
train_mask = np.ones(len(Y), bool)
train_mask[test_idx] = False
X_train, Y_train, X_test, Y_test = X[train_mask], Y[train_mask], X[test_idx], Y[test_idx]
print(f"X_train.shape: {X_train.shape}, X_test.shape: {X_test.shape}")

w = np.random.rand(3)
print(f"w.shape: {w.shape}")

accs = []
train_losses = []
valid_losses = []
preds = []
plt_cnt = 0

### Training and validation

def train():
    global plt_cnt
    for it in range(MAX_ITERS):
        kf = KFold(n_splits=N_SPLITS, shuffle=True)
        for train_idx, valid_idx in kf.split(X_train):
            X_train_fold, Y_train_fold = X_train[train_idx], Y_train[train_idx]
            X_valid_fold, Y_valid_fold = X_train[valid_idx], Y_train[valid_idx]
            for x, y in zip(X_train_fold, Y_train_fold):
                pred = np.dot(w[:2], x.T) + w[2]
                d = pred - y
                w[:2] -= LEARNING_RATE * d * x
                w[-1] -= LEARNING_RATE * d
                plt_cnt += 1
                # if plt_cnt % PLT_DELTA == 0:
                #     plot(X_train, Y_train, w)
                accs.append(
                    np.sum((w[0:2] @ X_valid_fold.T + w[-1] > 0.5) == Y_valid_fold.astype(bool)) / len(X_valid_fold)
                )
                train_losses.append(np.mean(np.abs(d)))
                valid_loss = np.mean(np.abs(np.dot(w[:2], X_valid_fold.T) + w[-1] - Y_valid_fold))
                if valid_loss < LOSS_THRESHOLD:
                    print(f"Early Stop when valid loss {valid_loss} less than threshold {LOSS_THRESHOLD}")
                    return
                valid_losses.append(
                    valid_loss
                )
train()

### Test dataset visualization

preds = (w[0:2] @ X_test.T + w[-1] > 0.5).astype(np.uint8)
print("Acc of test dataset:",np.mean(preds == Y_test))
cm = confusion_matrix(Y_test, preds)
plot_confusion_matrix(cm)
plot(X_test, Y_test, w)

### Accuracy visualization

fig = plt.figure()
plt.plot(accs)
plt.title("Accuracy")
plt.show()

### Loss visualization

fig = plt.figure()
plt.plot(train_losses, label="train_loss")
plt.plot(valid_losses, label="valid_loss")
plt.legend()
plt.title("losses")
plt.show()

