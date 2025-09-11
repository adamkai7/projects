from numpy import *
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn import tree
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import BaggingRegressor

from sklearn.tree import DecisionTreeRegressor,plot_tree
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_predict
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import RandomForestRegressor

def data_loader(fname):
    x_a = loadtxt(fname, skiprows=1, usecols=(1,2,3,4,5,7,8), delimiter=',', dtype=float64)
    y_v = loadtxt(fname, skiprows=1, usecols=0, delimiter=',', dtype=float64)

    return x_a, y_v


def tree_things(train_x_a, train_y_v):
    tree_reg = DecisionTreeRegressor(max_depth=None)
    tree_reg.fit(train_x_a, train_y_v)
    ytrain_pred = tree_reg.predict(train_x_a)
    train_mse = mean_squared_error(train_y_v, ytrain_pred)
    plt.figure(figsize=(20, 15))
    plot_tree(tree_reg, filled=True)
    plt.show()

    return tree_reg, train_mse

def crossval_for_tree(xtrain, ytrain, xtest, ytest, maxrange):
    train_mse_l = []
    test_mse_l = []
    cross_val_l = []
    for i in maxrange:
        tree_reg = DecisionTreeRegressor(max_depth=i)
        ypred = cross_val_predict(tree_reg, xtrain, ytrain, cv=50)
        cross_val_mse = mean_squared_error(ytrain, ypred)
        cross_val_l.append(cross_val_mse)

        tree_reg.fit(xtrain, ytrain)
        ytrainpred = tree_reg.predict(xtrain)
        trainmse = mean_squared_error(ytrain, ytrainpred)
        train_mse_l.append(trainmse)

        ytestpred = tree_reg.predict(xtest)
        testmse = mean_squared_error(ytest, ytestpred)
        test_mse_l.append(testmse)
    print("training mse", train_mse_l)
    print("test mse", test_mse_l)
    print("cross val mse", cross_val_l)
    plt.figure(figsize=(20, 15))
    plt.plot(maxrange, cross_val_l, marker='o', label='CV MSE')
    plt.plot(maxrange, train_mse_l, marker='o', label='Train MSE')
    plt.plot(maxrange, test_mse_l, marker='o', label='Test MSE')
    plt.xlabel('Max Depth')
    plt.ylabel('Mean Squared Error')
    plt.title('Cross-Validation, Training, Test')
    plt.legend()
    plt.show()

    return train_mse_l, test_mse_l, cross_val_l

def doin_bagging(xtrain, ytrain, xtest, ytest):
    ''' so basicaly do a tree then bagging?'''
    tree_reg = DecisionTreeRegressor()
    bag_reg = BaggingRegressor(base_estimator=tree_reg, n_estimators=1000)
    bag_reg.fit(xtrain, ytrain)
    ytestpred = bag_reg.predict(xtest)
    testmse = mean_squared_error(ytest, ytestpred)
    print("bagg mse", testmse)
    #test_mse, feature_importances = bagging_regression(xtrain, ytrain, xtest, ytest)

def forest_stuff(xtrain, ytrain, xtest, ytest, n_estimators=1000, max_features_values=None):
    teast_mse_l = []
    feature_importances = []

    for max_features in max_features_values:
        rf_reg = RandomForestRegressor(n_estimators=n_estimators, max_features=max_features, random_state=42)
        rf_reg.fit(xtrain, ytrain)
        y_test_pred = rf_reg.predict(xtest)
        test_mse = mean_squared_error(ytest, y_test_pred)
        teast_mse_l.append(test_mse)
        feature_importances.append(rf_reg.feature_importances_)



    plt.figure(figsize=(10, 6))
    plt.plot(max_features_values, teast_mse_l, marker='o')
    plt.xlabel('Max Features')
    plt.ylabel('Test Mean Squared Error')
    plt.title('Effect of Max Features on Test MSE')
    plt.show()
    for idx, max_features in enumerate(max_features_values):
        print(f"Max Features: {max_features}")
        print("Feature Importances:", feature_importances[idx])
        print()
    return teast_mse_l, feature_importances



def bart(train_x, train_y, test_x, test_y, num_trees=100, num_burn_in=100, num_iterations=1000):
    pass
    ''' note: never really used it before: heavily influenced from online tutotrials such as https://allenai.github.io/pybart/'''
    train_x = train_x.astype(float32)
    train_y = train_y.astype(float32)
    test_x = test_x.astype(float32)
    test_y = test_y.astype(float32)
    model = Model()
    model.num_trees = num_trees
    model.num_burn_in = num_burn_in
    model.num_iterations = num_iterations

    model.fit(train_x, train_y)

    y_test_pred = model.predict(test_x)

    test_mse = np.mean((test_y - y_test_pred) ** 2)
    print("Bart Test MSE:", test_mse)

    return test_mse



def main():
    x_a, y_v = data_loader("Carseats.csv")

    train_x_a = x_a[:300]
    train_y_v = y_v[:300]
    test_x_a = x_a[300:]
    test_y_v = y_v[300:]

#turn this back on later for decs tree pic
    tree_things(train_x_a, train_y_v)


    maxrange = range(1, 101)
    crossval_for_tree(train_x_a,train_y_v,test_x_a,test_y_v,maxrange)
    doin_bagging(train_x_a,train_y_v,test_x_a,test_y_v)
    max_features_values = [None, 0.2, 0.4, 0.6, 0.8]  
    teast_mse_, feature_importances = forest_stuff(train_x_a, train_y_v, test_x_a, test_y_v, max_features_values=max_features_values)
    #test_mse = bart(train_x_a, train_y_v, test_x_a, test_y_v)


if __name__ == '__main__':
    main()