import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_X_y, check_array#, check_is_fitted
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from mlxtend.evaluate import paired_ttest_5x2cv
from mlxtend.evaluate import mcnemar, mcnemar_table
from typing import Union

class LinearClassifier_(BaseEstimator):
    """
    | LinearClassifier_                                 |
    |---------------------------------------------------|
    | Own model programed to be compatible with sklearn,|
    |    based on the defined Linear Classifier in the  |
    |    report.                                        |
    |___________________________________________________|
    """
    def __init__(self, subestimator):
        """
        | __init__                                          |
        |---------------------------------------------------|
        | Class constructor.                                |
        |___________________________________________________|
        | sklearn.base ->  LinearClassifier_                |
        |___________________________________________________|
        | Input:                                            |
        | subestimator: sklearn model used to fit. In this  |
        |    case it will be a LinearRegression model.      |
        |___________________________________________________|
        | Output:                                           |
        | An object of the class.                           |
        """
        self.subestimator = subestimator
        self._estimator_type = "classifier"

    def fit(self, X, y):
        """
        | fit                                               |
        |---------------------------------------------------|
        | Function that performs the fitting process.       |
        |___________________________________________________|
        | ndarray, ndarray ->                               |
        |___________________________________________________|
        | Input:                                            |
        | X, y: ndarrays with train dataset values.         |
        |___________________________________________________|
        | Output:                                           |
        """
        X, y = check_X_y(X, y)

        self.subestimator.fit(X, y)

        return self

    def predict(self, X):
        """
        | predict                                           |
        |---------------------------------------------------|
        | Function that performs the prediction.            |
        |___________________________________________________|
        | ndarray, -> ndarray                               |
        |___________________________________________________|
        | Input:                                            |
        | X: ndarray with the test input values.            |
        |___________________________________________________|
        | Output:                                           |
        | ndarray with the prediction.                      |
        """
        X = check_array(X)

        y_pred = self.subestimator.predict(X)
        y_pred = np.clip(np.rint(y_pred), 0, 2).astype(np.int8)

        return y_pred


def load_data():
    """
    | load_data                                         |
    |---------------------------------------------------|
    | Function that performs the preprocessing.         |
    |___________________________________________________|
    |        ->                                         |
    |___________________________________________________|
    | Input:                                            |
    |___________________________________________________|
    | Output:                                           |
    | The X and y datasets.                             |
    """
    iris = datasets.load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = iris.target
    return X, y

def train_test_mcnemar(clf1, clf2, X:Union[list, np.ndarray], y:Union[list, np.ndarray], verbose:bool=True, test_size:float=0.25):
    """
    | train_test_mcnemar                                |
    |---------------------------------------------------|
    | Function that performs the train-test split, fit  |
    |    and prediction for McNemar test.               |
    |___________________________________________________|
    | sklearn.base, sklearn.base, ndarray, ndarray,     |
    |    bool, float -> ndarray, ndarray, ndarray       |
    |___________________________________________________|
    | Input:                                            |
    | clf1, clf2: sklearn type classification models to |
    |    be compared.                                   |
    | X, y: Iris dataset.                               |
    | verbose: flag that decide if the information      |
    |    should be displayed or no.                     |
    | test_size: percentaje size of the test set.       |
    |___________________________________________________|
    | Output:                                           |
    | The real and predicted by each model y value.     |
    """
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    # Fit
    clf1.fit(X_train, y_train)
    clf2.fit(X_train, y_train)
    # Predict
    y_m1 = clf1.predict(X_test)
    y_m2 = clf2.predict(X_test)

    if verbose:
        print(f"Predicted values by {clf1}: \n {y_m1}")
        print(f"Predicted values by {clf2}: \n {y_m2}")
        print(f"Real values: \n {y_test}")

    return y_test, y_m1, y_m2

def perform_mcnemar(y:Union[list, np.ndarray], y_m1:Union[list, np.ndarray], y_m2:Union[list, np.ndarray], verbose:bool=True):
    """
    | perform_mcnemar                                   |
    |---------------------------------------------------|
    | Function that performs the McNemar test using the |
    |    mlxtend pakage implementation.                 |
    |___________________________________________________|
    | ndarray, ndarray, ndarray, bool -> float, float   |
    |___________________________________________________|
    | Input:                                            |
    | y: the real values of the test dataset.           | 
    | y_m1, y_m2: predicted values for the test by the  |
    |    model 1 and 2 that are compared.               |
    | verbose: flag that decide if the information      |
    |    should be displayed or no.                     |
    |___________________________________________________|
    | Output:                                           |
    | The Chi2 and p-value obtained from the test.      |
    """
    table_to_test = mcnemar_table(y_target=y,
                                  y_model1=y_m1,
                                  y_model2=y_m2)
    if verbose:
        print(f'Tabla McNemar: \n {table_to_test}')
    
    chi2, p = mcnemar(ary=table_to_test, exact=False, corrected=True)

    if verbose:
        print(f'\n chi-squared: {chi2} \n p-value: {p}')

    return chi2, p

def perform_5x2cv_ttest(clf1, clf2, X:Union[list, np.ndarray], y:Union[list, np.ndarray], verbose:bool=True):
    """
    | perform_5x2cvttest                                |
    |---------------------------------------------------|
    | Function that performs the 5x2 cross-validation   |
    | t-test using the mlxtend pakage implementation.   |
    |___________________________________________________|
    | sklearn.base, sklearn.base, ndarray, ndarray,     |
    |    bool -> float, float                           |
    |___________________________________________________|
    | Input:                                            |
    | clf1, clf2: sklearn type classification models to |
    |    be compared.                                   |
    | X, y: Iris dataset.                               |
    |    model 1 and 2 that are compared.               |
    | verbose: flag that decide if the information      |
    |    should be displayed or no.                     |
    |___________________________________________________|
    | Output:                                           |
    | The t and p-value obtained from the test.         |
    """
    t, p = paired_ttest_5x2cv(estimator1=clf1,
                              estimator2=clf2,
                              X=X, 
                              y=y)
    if verbose:
        print(f'\n t statistic: {t} \n p-value: {p}')
    
    return t, p

def loop_steps(steps:int=5, file:str='./out.csv'):
    """
    | loop_steps                                        |
    |---------------------------------------------------|
    | Function that control the comparison of the tests |
    |    given the predefined models, in terms of where |
    |    to save the results and how many repetition of |
    |    the comparison to perfom.                      |
    |___________________________________________________|
    | int, str ->                                       |
    |___________________________________________________|
    | Input:                                            |
    | steps: amount of repetition for the loop.         |
    | file: where to save the results.                  |
    |___________________________________________________|
    | Output:                                           |
    | Nothing, the result is saved in the given file.   |
    """
    res_list = []
    for i in range(steps):
        # Build classifiers
        lc = LinearClassifier_(subestimator=LinearRegression())
        knn = KNeighborsClassifier(n_neighbors=3)

        # Load data
        X, y = load_data()

        # Train and predict for McNemar
        y_test, y_m1, y_m2 = train_test_mcnemar(lc, knn, X, y)

        # Perform McNemar
        chi2, p_mc = perform_mcnemar(y_test, y_m1, y_m2)

        # Perform 5x2cv T-Test
        t, p_tt = perform_5x2cv_ttest(lc, knn, X, y)

        res_list.append([chi2, p_mc, t, p_tt])
    
    df = pd.DataFrame(res_list, columns=['Chi2', 'p_McNemar', 't', 'p_5x2cvTtest'])
    df.to_csv(file)


def main():
    loop_steps(10)

if __name__ == "__main__":
    main()