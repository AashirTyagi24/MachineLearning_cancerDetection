import pandas as pd
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score
from gaussian import GaussianGenerativeModel
import matplotlib.pyplot as plt


np.random.seed(0)

# read data from csv file
df = pd.read_csv("data.csv")
df['diagnosis'] = pd.Series(np.where(df.diagnosis.values == 'B', 1, 0), df.index)

for column in df.columns:
    if column != 'diagnosis':
        _mean = df[column].mean()
        df[column].fillna(_mean, inplace=True)

shuffled_df = df.sample(frac=1, axis=1)
# split data into training and testing sets
num_splits = 50
train_frac = 0.67
test_frac = 1 - train_frac
results1 = []
results2 = []
for i in range(num_splits):
    print('Iteration #{}'.format(i + 1))
    # randomly split the data
    train_df = df.sample(frac=train_frac)
    test_df = df.drop(train_df.index)

    shuffled_train_df = shuffled_df.sample(frac=train_frac)
    shuffled_test_df = shuffled_df.drop(shuffled_train_df.index)
    
    
    # split data into input features and target variable
    X_train = train_df.drop('diagnosis', axis=1)
    y_train = train_df['diagnosis']
    y_train = y_train.astype('int')
    X_test = test_df.drop('diagnosis', axis=1)
    y_test = test_df['diagnosis'].astype('int')
    
    # Fisher's Linear Discriminant Model
    FLDM1 = LinearDiscriminantAnalysis(n_components=1)
    X_train_lda = FLDM1.fit_transform(X_train, y_train) 
    X_test_lda = FLDM1.transform(X_test)

    model = GaussianGenerativeModel()
    model.fit(X_train_lda, y_train)
    y_pred = model.predict(X_test_lda)
    accuracy = accuracy_score(y_test, y_pred)
    results1.append(accuracy)

    sigma_inv = np.linalg.inv(model.sigma)
    w = sigma_inv @ (model.mu_1 - model.mu_0)
    w0 = -0.5 * (model.mu_1.T @ sigma_inv @ model.mu_1 - model.mu_0.T @ sigma_inv @ model.mu_0) + np.log(model.phi / (1 - model.phi))

    # plot decision boundary
    if len(w) > 1:
        x1 = np.linspace(X_train_lda[:, 0].min(), X_train_lda[:, 0].max(), 100)
        x2 = (-w0 - w[0] * x1) / w[1]
        plt.plot(x1, x2, 'k--', label='Decision boundary')
    else:
        x = np.linspace(X_train_lda[:, 0].min(), X_train_lda[:, 0].max(), 100)
        plt.axvline(x=w0 / w[0], color='k', linestyle='--', label='Decision boundary')
    plt.scatter(X_train_lda[:, 0], y_train, c=y_train, cmap='bwr', alpha=0.5)
    plt.xlabel('LD1')
    plt.legend()
    plt.savefig(f'plots/FLDM1-{i}.png') 
    plt.clf()

    # split data into input features and target variable
    X_train_shuffle = shuffled_train_df.drop('diagnosis', axis=1)
    y_train_shuffle = shuffled_train_df['diagnosis']
    y_train_shuffle = y_train_shuffle.astype('int')
    X_test_shuffle = shuffled_test_df.drop('diagnosis', axis=1)
    y_test_shuffle = shuffled_test_df['diagnosis'].astype('int')

    FLDM2 = LinearDiscriminantAnalysis(n_components=1)
    X_train_shuffle_lda = FLDM2.fit_transform(X_train_shuffle, y_train_shuffle) 
    X_test_shuffle_lda = FLDM2.transform(X_test_shuffle)
    # thresholds1.append(FLDM2.threshold_)

    model.fit(X_train_shuffle_lda, y_train_shuffle)
    y_pred = model.predict(X_test_shuffle_lda)
    accuracy = accuracy_score(y_test_shuffle, y_pred)
    results2.append(accuracy)

    sigma_inv = np.linalg.inv(model.sigma)
    w = sigma_inv @ (model.mu_1 - model.mu_0)
    w0 = -0.5 * (model.mu_1.T @ sigma_inv @ model.mu_1 - model.mu_0.T @ sigma_inv @ model.mu_0) + np.log(model.phi / (1 - model.phi))

    # plot decision boundary
    if len(w) > 1:
        x1 = np.linspace(X_train_shuffle_lda[:, 0].min(), X_train_shuffle_lda[:, 0].max(), 100)
        x2 = (-w0 - w[0] * x1) / w[1]
        plt.plot(x1, x2, 'k--', label='Decision boundary')
    else:
        x = np.linspace(X_train_shuffle_lda[:, 0].min(), X_train_shuffle_lda[:, 0].max(), 100)
        plt.axvline(x=w0 / w[0], color='k', linestyle='--', label='Decision boundary')
    plt.scatter(X_train_shuffle_lda[:, 0], y_train_shuffle, c=y_train_shuffle, cmap='bwr', alpha=0.5)
    plt.xlabel('LD1')
    plt.legend()
    plt.savefig(f'plots/FLDM2-{i}.png') 
    plt.clf()
    
    # train and evaluate model

# print results
print("Results of", num_splits, "random training/testing splits:")
print('FLDM1')
print("Average accuracy:", np.mean(results1))
print("Accuracy variance:", np.var(results1))
print('FLDM2')
print("Average accuracy:", np.mean(results2))
print("Accuracy variance:", np.var(results2))