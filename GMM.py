# ===============================================================
# 📘 Gaussian vs Gaussian Mixture Model (MNIST ROC Visualization)
# ===============================================================
# ✅ Keeps all model behavior and outputs identical
# ✅ Plots per-class ROC comparison (Gaussian vs GMM)
# ✅ Auto-zooms each plot so curves are in-frame
# ===============================================================

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from scipy.stats import multivariate_normal as mvn
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.stats import multivariate_normal

# ---------------------------------------------------------------
# 🧩 Gaussian Model Functions
# ---------------------------------------------------------------
def TrainGaussianModel(X_train, y_train):
    class_labels = np.unique(y_train)
    mio, cov = [], []
    for c in class_labels:
        cdata = X_train[np.where(y_train == c)].T
        mio.append(np.mean(cdata, 1))
        cov.append(np.cov(cdata))
    return class_labels, mio, cov

def TestGaussianModel(X_test, class_labels, mio, cov):
    m, nc = X_test.shape[0], len(class_labels)
    LL = np.zeros((m, nc))
    for i in range(nc):
        LL[:, i] = mvn.logpdf(X_test, mio[i], cov[i])
    indices = np.argmax(LL, axis=1)
    y_pred = np.array([class_labels[i] for i in indices])
    return y_pred, LL

# ---------------------------------------------------------------
# 🧩 GMM Functions
# ---------------------------------------------------------------
def init(X, K):
    alpha = np.random.uniform(0, 1, K)
    alpha /= np.sum(alpha)
    mean = np.mean(X, 1)
    cov = np.cov(X)
    gm = [mvn(mean, cov) for _ in range(K)]
    return alpha, gm

def CompProbabilities(X, alpha, gm):
    m, K = X.shape[1], len(alpha)
    gamma = np.zeros((m, K))
    for k in range(K):
        for i in range(m):
            LP = np.log(alpha[k]) + gm[k].logpdf(X[:, i])
            gamma[i, k] = LP
    max_gamma = np.max(gamma)
    gamma = np.exp(gamma - max_gamma)
    gamma[gamma < 1e-300] = 1e-300
    sum_gamma_cols = np.sum(gamma, 1)
    for i in range(m):
        gamma[i, :] /= sum_gamma_cols[i]
    P = np.sum(np.log(sum_gamma_cols)) + m * max_gamma
    return gamma, P

def ReestimateParameters(X, gamma, gm, best_gm):
    n, m = X.shape
    K = gamma.shape[1]
    alpha = np.sum(gamma, 0)
    for k in range(K):
        mio = np.zeros(n)
        for i in range(m):
            mio += gamma[i, k] * X[:, i]
        mio /= (alpha[k] + 1e-8)
        segma = np.zeros((n, n))
        for i in range(m):
            X1 = np.reshape(X[:, i] - mio, (n, 1))
            segma += gamma[i, k] * (X1 @ X1.T)
        segma /= (alpha[k] + 1e-8)
        segma += np.eye(n) * 1e-6
        try:
            gm[k] = multivariate_normal(mio, segma, allow_singular=True)
        except np.linalg.LinAlgError:
            gm[k] = best_gm[k]
    alpha /= m
    return alpha, gm

def TrainGaussianMixtureOfClass(X, K):
    n, m = X.shape
    alpha, gm = init(X, K)
    PP = -np.inf
    best_alpha, best_gm = alpha, gm
    for t in range(30):
        gamma, P = CompProbabilities(X, alpha, gm)
        if P > PP:
            best_alpha, best_gm, PP = alpha.copy(), gm.copy(), P
        alpha, gm = ReestimateParameters(X, gamma, gm, best_gm)
    return best_alpha, best_gm, PP

def TrainGaussianMixtureModel(X_train, y_train, K):
    class_labels = np.unique(y_train)
    alphas, gms = [], []
    for c in class_labels:
        cdata = X_train[np.where(y_train == c)].T
        alpha, gm, P = TrainGaussianMixtureOfClass(cdata, K)
        alphas.append(alpha)
        gms.append(gm)
    return class_labels, alphas, gms

def TestGaussianMixtureModel(X_test, class_labels, alphas, gms):
    m, nc = X_test.shape[0], len(class_labels)
    K = alphas[0].shape[0]
    LL = np.zeros((m, nc))
    for i in range(m):
        for c in range(nc):
            for k in range(K):
                LL[i, c] += alphas[c][k] * gms[c][k].pdf(X_test[i, :])
    indices = np.argmax(LL, axis=1)
    y_pred = np.array([class_labels[i] for i in indices])
    return y_pred, LL

# ---------------------------------------------------------------
# 📊 ROC Function
# ---------------------------------------------------------------
def compute_ROC(y_true, scores, class_label):
    y_bin = (y_true == class_label).astype(int)
    thresholds = np.sort(np.unique(scores))
    Pd, Pf = [], []
    n1, n0 = np.sum(y_bin == 1), np.sum(y_bin == 0)
    for tau in thresholds:
        y_pred = (scores >= tau).astype(int)
        TP = np.sum((y_pred == 1) & (y_bin == 1))
        FP = np.sum((y_pred == 1) & (y_bin == 0))
        Pd.append(TP / n1)
        Pf.append(FP / n0)
    return np.array(Pf), np.array(Pd)

# ---------------------------------------------------------------
# 🧠 Main Experiment
# ---------------------------------------------------------------
mnist = fetch_openml('mnist_784', version=1, as_frame=False)
X, y = mnist.data, mnist.target.astype(int)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# PCA for speed
pca = PCA(n_components=40)
pca.fit(X_train)
X_train, X_test = pca.transform(X_train), pca.transform(X_test)

# Train Gaussian Model
print("Training Gaussian Model...")
class_labels, mio, cov = TrainGaussianModel(X_train, y_train)
y_pred_gauss, LL_gauss = TestGaussianModel(X_test, class_labels, mio, cov)
acc_gauss = 1 - np.mean(y_pred_gauss != y_test)
print("Gaussian Model Accuracy =", acc_gauss)

# Train GMM Model
print("Training Gaussian Mixture Model...")
class_labels, alphas, gms = TrainGaussianMixtureModel(X_train, y_train, K=5)
y_pred_gmm, LL_gmm = TestGaussianMixtureModel(X_test, class_labels, alphas, gms)
acc_gmm = 1 - np.mean(y_pred_gmm != y_test)
print("Gaussian Mixture Model Accuracy =", acc_gmm)

# ---------------------------------------------------------------
# 📈 ROC Curves per class (saved individually)
# ---------------------------------------------------------------
output_dir = "ROC_Individual_Plots"
os.makedirs(output_dir, exist_ok=True)

for i, c in enumerate(class_labels):
    Pf_gauss, Pd_gauss = compute_ROC(y_test, LL_gauss[:, i], c)
    Pf_gmm, Pd_gmm = compute_ROC(y_test, LL_gmm[:, i], c)

    plt.figure(figsize=(4, 3))
    plt.plot(Pf_gauss, Pd_gauss, color='blue', lw=2, label='Gaussian Model')
    plt.plot(Pf_gmm, Pd_gmm, color='red', lw=2, label='GMM Model')
    plt.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5, label='Random Baseline')

    # 🧭 Auto-zoom: ensures curves are visible
    plt.xlim([0, min(1, max(Pf_gauss.max(), Pf_gmm.max()) * 1.1)])
    plt.ylim([max(0, min(Pd_gauss.min(), Pd_gmm.min()) - 0.05), 1.0])

    plt.xlabel('False Positive Rate (Pfa)', fontsize=6, fontweight='bold', color='blue')
    plt.ylabel('True Positive Rate (Pd)', fontsize=6, fontweight='bold', color='red')
    plt.title(f'ROC Curve - Class {c}', fontsize=8, fontweight='bold', color='red')
    plt.legend(loc='lower right', fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(os.path.join(output_dir, f"ROC_Class_{c}.png"), dpi=300)
    plt.close()

print(f"✅ Individual ROC PNGs saved in folder: {output_dir}")
