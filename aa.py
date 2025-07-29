import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
import seaborn as sns
import numpy.random as nr #for random seed
from sklearn.preprocessing import StandardScaler
from sklearn import feature_selection as fs
from sklearn import model_selection as ms
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from statsmodels.stats.outliers_influence import variance_inflation_factor
from math import sqrt
import warnings
warnings.filterwarnings('ignore')
import random
#for maintaining randomness
random_state = 1
# define file name
file_name = "Dataset.csv"
# read file from csv to pandas DataFrame
df = pd.read_csv(file_name)
df
plt.figure(figsize=(12, 6))
plt.subplot(2, 2, 1)
sns.histplot(data=df, x='co2', bins=20, kde=True)
plt.title('CO2 Emissions Distribution')
plt.subplot(2, 2, 2)
sns.histplot(data=df, x='population', bins=20, kde=True)
plt.title('Population Distribution')
plt.subplot(2, 2, 3)
sns.histplot(data=df, x='gdp', bins=20, kde=True)
plt.title('GDP Distribution')
plt.subplot(2, 2, 4)
sns.histplot(data=df, x='primary_energy_consumption', bins=20, kde=True)
plt.title('Primary Energy Consumption Distribution')
plt.tight_layout()
plt.show()
# Correlation Matrix Heatmap
correlation_matrix = df.select_dtypes(include=[np.number]).corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix Heatmap')
plt.show()
# Bar Plot: Top Countries by CO2 Emissions
top_countries = df.groupby('country')['co2'].sum().sort_values(ascending=False).head(10)
plt.figure(figsize=(10, 6))
sns.barplot(x=top_countries.values, y=top_countries.index, palette='viridis')
plt.title('Top Countries by Total CO2 Emissions')
plt.xlabel('Total CO2 Emissions')
plt.ylabel('Country')
plt.tight_layout()
plt.show()
# #  Handling outlier from previous analysis
#Select relevant features from previous analysis
final_data = df[['country','year','co2','coal_co2','cement_co2','gas_co2','oil_co2','methane','population','gdp']]
#Remove Outliers (countries) with significantly  high range features
final_data = final_data[final_data['country'].isin(['Afghanistan', 'Albania', 'Algeria', 'Argentina', 'Armenia',
'Australia', 'Austria', 'Azerbaijan', 'Belarus', 'Belgium',
'Benin', 'Bolivia', 'Bosnia and Herzegovina', 'Botswana',
'Bulgaria', 'Cameroon', 'Canada', 'Chile', 'Colombia', 'Croatia',
'Cuba', 'Cyprus', 'Czechia', 'Denmark', 'Dominican Republic',
'Egypt', 'Estonia', 'Finland', 'France', 'Georgia', 'Ghana',
'Greece', 'Guatemala', 'Hungary', 'Iceland', 'Iraq', 'Ireland',
'Israel', 'Italy', 'Jamaica', 'Jordan', 'Kazakhstan', 'Kyrgyzstan',
'Latvia', 'Lebanon', 'Libya', 'Lithuania', 'Luxembourg',
'Malaysia', 'Mexico', 'Moldova', 'Morocco', 'Mozambique',
'Netherlands', 'New Zealand', 'North Macedonia', 'Norway',
'Panama', 'Peru', 'Philippines', 'Poland', 'Portugal', 'Romania',
'Rwanda', 'Senegal', 'Serbia', 'Slovakia', 'Slovenia',
'South Korea', 'Spain', 'Sweden', 'Switzerland', 'Syria',
'Tajikistan', 'Tanzania', 'Thailand', 'Tunisia', 'Turkey',
'Turkmenistan', 'Ukraine', 'United Arab Emirates',
'United Kingdom', 'Uruguay', 'Uzbekistan', 'Venezuela', 'Yemen'])]
# # Data Overview
final_data.shape
print('Columns and thier data types')
final_data.dtypes
print('Data preview')
final_data.head()
print('data statistics:')
final_data.describe()
print('A pairplot of selected features before dimention reduction')
sns.pairplot(final_data[['co2', 'coal_co2', 'cement_co2', 'gas_co2', 'oil_co2', 'methane', 'population', 'gdp']],
diag_kind="kde")
#dimensionality reduction
final_data['ccgo'] = final_data['cement_co2'] + final_data['gas_co2'] + final_data['oil_co2'] + final_data['coal_co2']
final_data['gdp_per_capita'] = final_data['gdp'] / final_data['population']
final_data.head()
data = final_data.drop(['cement_co2','gas_co2','oil_co2','coal_co2','gdp','population'],axis=1)
data.head()
data.year.unique()
# #  Splitting the dataset into training and test data
ft_cols = ['year','methane','ccgo','gdp_per_capita']
lb_col = ['co2']
features = np.array(data[ft_cols])
label = np.array(data[lb_col]).ravel()
#Data splitting using sklearn train_test_split function
ft_train,ft_test,lb_train,lb_test = ms.train_test_split(features,label,test_size=0.3
,shuffle = True, random_state= random_state)
scaler = StandardScaler()
scaler.fit(ft_train)
ft_train_s = scaler.fit_transform(ft_train)
ft_test_s = scaler.transform(ft_test)
# # Using KNN
nr.seed(5)
KNR = KNeighborsRegressor()
k_parameters = {'n_neighbors': np.arange(1,40,1)}
KNR_cv_model = ms.GridSearchCV(KNR, k_parameters, cv = 5)
#Fitting the KNR model to the dataset
KNR_cv_model.fit(ft_train_s, lb_train)

#Best parameters
KNR_cv_model.best_params_
# Tuned model
KNR_tuned = KNR_cv_model.best_estimator_
# plot predicted vs actual values
predictions = KNR_tuned.predict(ft_test_s)
f,ax=plt.subplots(figsize=(20,15))
sns.set(font_scale=2)
sns.regplot(x=predictions, y=np.transpose(lb_test), fit_reg=True)
plt.xlabel("CO2 emissions [Mt] - predicted")
plt.ylabel("CO2 emissions [Mt] - actual")
plt.title("Pearsons Correlation coefficient R= {}".format(round(np.corrcoef(predictions,np.transpose(lb_test))[0,1],2)))
plt.show()
# Model Evaluation
test = lb_test
predictions = KNR_tuned.predict(ft_test_s)
print('Tuned KNeighborsRegressor')
KNN_MAE = mean_absolute_error(test, predictions)
print('MAE : {}'.format(KNN_MAE))
KNN_MSE = mean_squared_error(test, predictions)
print('MSE : {}'.format(KNN_MSE))
KNN_RMSE = sqrt(KNN_MSE)
print('RMSE : %f' % KNN_RMSE)
KNN_R2_score = r2_score(test, predictions)
print('R2_score : {}'.format(KNN_R2_score))
frame = pd.DataFrame()
frame['test'] = test
frame['predictions'] = predictions
# # Using Random Forest
RFR = RandomForestRegressor(random_state = 42)
#RFR_params = {'max_depth': list(range(1,10)),'max_features': [3],'n_estimators' : [10,20,30,40]}
# Getting the best parameter using Grid Search
#RFR_model = ms.GridSearchCV(RFR,RFR_params,cv = 5, n_jobs = -1,verbose = 2)
RFR.fit(ft_train_s, lb_train)
#Best parameters
#RFR_model.best_params_
#RFR_tuned = RFR_model.best_estimator_
# plot predicted vs actual values
predictions1 = RFR.predict(ft_test_s)
f,ax=plt.subplots(figsize=(20,15))
sns.set(font_scale=2)
sns.regplot(x=predictions1, y=np.transpose(lb_test), fit_reg=True)
plt.xlabel("CO2 emissions [MT] - predicted")
plt.ylabel("CO2 emissions [MT] - actual")
plt.title("Pearsons Correlation coefficient R= {}".format(round(np.corrcoef(predictions1,np.transpose(lb_test))[0,1],2)))
plt.show()
# Model Evaluation
test = lb_test
predictions1 = RFR.predict(ft_test_s)
print('Tuned Random Forest')
RF_MAE = mean_absolute_error(test, predictions1)
print('MAE : {}'.format(RF_MAE))
RF_MSE = mean_squared_error(test, predictions1)
print('MSE : {}'.format(RF_MSE))
RF_RMSE = sqrt(RF_MSE)
print('RMSE : %f' % RF_RMSE)
RF_R2_score = r2_score(test, predictions1)
print('R2_score : {}'.format(RF_R2_score))
frame = pd.DataFrame()
frame['test'] = test
frame['predictions'] = predictions1
#Summary of model scores from different metrics
KNN_model = KNN_MAE,KNN_MSE,KNN_RMSE,KNN_R2_score
RF_model = RF_MAE,RF_MSE,RF_RMSE,RF_R2_score
summary = pd.DataFrame([KNN_model,RF_model],
index = ['KNN','RF'],
columns = ['MAE','MSE','RMSE','R2_score'])
summary
plt.figure(figsize=(15,10))
plt.barh(width = summary['MAE'], y = summary.index)
plt.xlabel("Mean Absolute Error [MT]")
plt.ylabel("Models")
plt.title("Mean absolute error of models [MT]")
plt.figure(figsize=(15,10))
plt.barh(width = summary['MSE'], y = summary.index)
plt.xlabel("Mean Squared Error [MT]")
plt.ylabel("Models")
plt.title("Mean Squared Error of models [MT]")
plt.figure(figsize=(15,10))
plt.barh(width = summary['RMSE'], y = summary.index)
plt.xlabel("Root Mean Squared Error [MT]")
plt.ylabel("Models")
plt.title("Root Mean Squared Error of models [MT]")
plt.figure(figsize=(15,10))
plt.barh(width = summary['R2_score'], y = summary.index)
plt.xlabel("R2 Score of models")
plt.ylabel("Models")
plt.title("Accuracy of models") 
