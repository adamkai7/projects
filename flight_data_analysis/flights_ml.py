from numpy import *
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

'''
created by Adam Kainikara

contact: adamkainikara@gmail.com

Analyzed U.S. flight data to classify delayed vs. on-time flights and predict arrival delays using Random Forest and Linear Regression. 
'''
# dtype for flight data for this project arrival delay, departure delay, distance traveled and air time will be analyzed
dt = dtype([
    ('arr_delay', float),
    ('dep_delay', float),
    ('distance', float),
    ('air_time', float)
])

from sklearn.model_selection import train_test_split

def load_data(path):
    # this loads flights using loadtxt because loadtxt can deal with missing values
    data_a = genfromtxt(path, delimiter=',', skip_header=1, usecols=(9,6,18,16), dtype=dt, invalid_raise=False)
    # remove rows with NaNs there are some that are missing data in a particular column
    mask_v = ~any(isnan(data_a.view(float).reshape(data_a.shape + (-1,))), axis=1)
    # mask of the rows to remove
    data_a = data_a[mask_v]
    return data_a


# distance delay correlation
def distance_corr(arr_delay_v, distance_v):
    return corrcoef(distance_v, arr_delay_v)[0, 1]
    # computes the correlation, in this case we are investigating the correlation between distance traveled and arrival delay

# classification on time vs delayed
def classify(x_a, y_v):
    xtr_a, xte_a, ytr_v, yte_v = train_test_split(x_a, y_v, test_size=0.3, random_state=0)
    # training and test data 
    clf = RandomForestClassifier(n_estimators=100, random_state=0)
    clf.fit(xtr_a, ytr_v)
    # fitting and predicting
    yp_v = clf.predict(xte_a)
    print('classification accuracy:', accuracy_score(yte_v, yp_v))
    print(classification_report(yte_v, yp_v))

# regression predict arrival delay
def regress(x_a, y_v):
    xtr_a, xte_a, ytr_v, yte_v = train_test_split(x_a, y_v, test_size=0.3, random_state=0)
        # training and test data 

    lr = LinearRegression()
    lr.fit(xtr_a, ytr_v)
    yp_v = lr.predict(xte_a)
    print('linear regression rmse:', sqrt(mean_squared_error(yte_v, yp_v)))
        # fitting and predicting both linear regression and random forest

    rf = RandomForestRegressor(n_estimators=100, random_state=0)
    rf.fit(xtr_a, ytr_v)
    yp2_v = rf.predict(xte_a)
    print('random forest rmse:', sqrt(mean_squared_error(yte_v, yp2_v)))





#  visualization functions 

def plot_arrival_delay_distribution(arr_delay_v):
    #histogram of arrival delays with classification threshold
    plt.figure(figsize=(8,5))
    sns.histplot(arr_delay_v, bins=50, color='skyblue')
    plt.axvline(15, color='red', linestyle='--', label='delayed threshold (15 min)')
    plt.xlabel('arrival delay (minutes)')
    plt.ylabel('number of flights')
    plt.title('arrival delay distribution')
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_on_time_vs_delayed(arr_delay_v):
    #pie chart showing ratio of on-time vs delayed flights
    delayed_v = (arr_delay_v > 15).astype(int)
    counts_v = [sum(delayed_v == 0), sum(delayed_v == 1)]
    labels_v = ['on-time', 'delayed']
    colors_v = ['green', 'red']
    
    plt.figure(figsize=(6,6))
    plt.pie(counts_v, labels=labels_v, autopct='%1.1f%%', colors=colors_v, startangle=90)
    plt.title('proportion of on-time vs delayed flights')
    plt.tight_layout()
    plt.show()


def plot_distance_vs_delay(distance_v, arr_delay_v):
    #scatter plot showing correlation between distance and arrival delay
    plt.figure(figsize=(8,5))
    sns.scatterplot(x=distance_v, y=arr_delay_v, alpha=0.3)
    plt.xlabel('distance (miles)')
    plt.ylabel('arrival delay (minutes)')
    plt.title(f'distance vs arrival delay (corr={corrcoef(distance_v, arr_delay_v)[0,1]:.3f})')
    plt.tight_layout()
    plt.show()


def plot_regression_errors(y_true_v, y_pred_v, model_name='model'):
    #histogram of regression prediction errors
    errors_v = y_pred_v - y_true_v
    plt.figure(figsize=(8,5))
    sns.histplot(errors_v, bins=50, color='orange')
    plt.xlabel('prediction error (minutes)')
    plt.ylabel('number of flights')
    plt.title(f'{model_name} prediction errors')
    plt.axvline(0, color='black', linestyle='--')
    plt.tight_layout()
    plt.show()


def plot_feature_importance(x_a, y_v, feature_names_v):
    #bar chart of feature importance from randomforestclassifier
    clf = RandomForestClassifier(n_estimators=100, random_state=0)
    clf.fit(x_a, y_v)
    importances_v = clf.feature_importances_
    
    plt.figure(figsize=(6,4))
    sns.barplot(x=importances_v, y=feature_names_v, palette='viridis')
    plt.xlabel('importance')
    plt.ylabel('feature')
    plt.title('random forest feature importance')
    plt.tight_layout()
    plt.show()


def main():
    data_a = load_data('flights.csv')
    
    # extract vectors/arrays
    arr_delay_v = data_a['arr_delay']
    dep_delay_v = data_a['dep_delay']
    distance_v = data_a['distance']
    air_time_v = data_a['air_time']
    
    # prepare input array
    x_a = column_stack([dep_delay_v, distance_v, air_time_v])
    feature_names_v = ['dep_delay', 'distance', 'air_time']
    
    # classification target
    y_class_v = (arr_delay_v > 15).astype(int)
    
    # print correlation
    print('corr distance vs arr_delay:', corrcoef(distance_v, arr_delay_v)[0, 1])
    
    # run models
    classify(x_a, y_class_v)
    regress(x_a, arr_delay_v)
    
    #  call visualizations 
    plot_arrival_delay_distribution(arr_delay_v)
    plot_on_time_vs_delayed(arr_delay_v)
    plot_distance_vs_delay(distance_v, arr_delay_v)
    
    # regression errors
   
    
    xtr_a, xte_a, ytr_v, yte_v = train_test_split(x_a, arr_delay_v, test_size=0.3, random_state=0)
    
    # linear regression
    lr = LinearRegression()
    lr.fit(xtr_a, ytr_v)
    yp_lr_v = lr.predict(xte_a)
    plot_regression_errors(yte_v, yp_lr_v, model_name='linear regression')
    
    # random forest regression
    rf = RandomForestRegressor(n_estimators=100, random_state=0)
    rf.fit(xtr_a, ytr_v)
    yp_rf_v = rf.predict(xte_a)
    plot_regression_errors(yte_v, yp_rf_v, model_name='random forest regression')
    
    # feature importance
    plot_feature_importance(x_a, y_class_v, feature_names_v)


if __name__ == '__main__':
    main()
