from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor

from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline


def train_model(model_name, X_train, y_train, X_test, y_test):
    if model_name == 'linear':
        # Linear regression
        clf = LinearRegression(n_jobs=-1)
        clf.fit(X_train, y_train)

    elif model_name == 'quadratic2':
        # Quadratic Regression 2
        clf = make_pipeline(PolynomialFeatures(2), Ridge())
        clf.fit(X_train, y_train)

    elif model_name == 'quadratic3':
        # Quadratic Regression 3
        clf = make_pipeline(PolynomialFeatures(3), Ridge())
        clf.fit(X_train, y_train)

    elif model_name == 'knn':
        # KNN Regression
        clf = KNeighborsRegressor(n_neighbors=2)
        clf.fit(X_train, y_train)

    return clf

    #confidence = clf.score(X_test, y_test)
    #print('Model "{}" confidence is {}'.format(model_name, confidence))
