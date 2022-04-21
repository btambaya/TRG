#!/usr/bin/env python
# coding: utf-8

# ## 1.Libraries

# In[1]:


import pandas as pd
import numpy as np
from numpy import mean, std
import seaborn as sns 
import matplotlib.pyplot as plt
from matplotlib import pyplot
from matplotlib.gridspec import GridSpec

from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestRegressor
from catboost import Pool, CatBoostRegressor, cv
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import StackingRegressor


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)

execfile('viz.py')


# ## 2. Reading the Data

# In[2]:


data = pd.read_csv('Data.csv')
data.head(5)


# ### 2.1 An Overview from the Data

# In[3]:


print('Total Amount of Duplicates: ',data.duplicated().sum())


# In[4]:


data.info()


# In[5]:


total = data.isnull().sum().sort_values(ascending = False)
percent = (data.isnull().sum()/data.isnull().count()*100).sort_values(ascending = False)
missing_data  = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data


# In[6]:


data.describe()


# In[7]:


data.describe(include='O') 


# ## 3.EDA

# So now we will go trough an exploratory data analysis to get insights from the first week data of Register users. The aim here is to divide this session into topics so we can explore graphics for each subject.
# 1. Transform timestamp columns
# 2. Extract time attributes from these datetime columns (year, month and day)
# 3. Evaluate the scenario using this attributes

# In[8]:


data["DateFormatted"]=data["DateFormatted"].str.replace("/","-")
data["DateFormatted"] = pd.to_datetime(data["DateFormatted"])


# In[9]:


# Extracting attributes for Registration date - Year, Month and Day
data['registration_year'] = data["DateFormatted"].apply(lambda x: x.year)
data['registration_monthID'] = data["DateFormatted"].apply(lambda x: x.month)
data['registration_month'] = data["DateFormatted"].apply(lambda x: x.strftime('%b'))
data['registration_year_month'] = data["DateFormatted"].apply(lambda x: x.strftime('%Y%m'))
data['registration_day'] = data["DateFormatted"].apply(lambda x: x.strftime('%a'))
data["DateFormatted"] = data["DateFormatted"].apply(lambda x: x.strftime('%Y%m%d'))


# In[10]:


# plotting a chart for Registration year
fig = plt.figure(constrained_layout=True, figsize=(20, 10))
gs = GridSpec(2, 2, figure=fig)
ax1 = fig.add_subplot(gs[0, 0])

single_countplot(data, x='registration_year', ax=ax1,  palette='magma')
ax1.set_title('registration_year', size=12, color='dimgrey')

plt.show()


# by the chart above displaying the distribution of registration carried out between 2017 to 2019, it can be said that registration of new customers has be quite constant over the years.

# In[11]:


fig = plt.figure(figsize=(13, 20))

# Axis definition
gs = GridSpec(3, 3, figure=fig)
ax1 = fig.add_subplot(gs[0, :])
ax2 = fig.add_subplot(gs[1, :])
ax3 = fig.add_subplot(gs[2, :])

# Lineplot - Evolution of Registration 
sns.lineplot(data=data['registration_year_month'].value_counts().sort_index(), ax=ax1, 
             color='darkslateblue', linewidth=2)
format_spines(ax1, right_border=False)  
for tick in ax1.get_xticklabels():
    tick.set_rotation(45)
ax1.set_title('Evolution of Registration', size=14, color='dimgrey')

# Barchart - Total Registration by Day of Week
single_countplot(data, x='registration_day', ax=ax2, order=False, palette='YlGnBu')
weekday_label = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
ax2.set_xticklabels(weekday_label)
ax2.set_title('Total Registration by Day of Week', size=14, color='dimgrey', pad=20)

# Bar chart - Registration Comparison between 2017, 2018 and 2019
df_orders_compare =data.query('registration_year in (2017, 2018, 2019) & registration_monthID <= 12')
single_countplot(df_orders_compare, x='registration_monthID', hue='registration_year', ax=ax3, order=False,
                 palette='YlGnBu')
month_label = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug','Sept','Oct','Nov','Dec']
ax3.set_xticklabels(month_label)
ax3.set_title('Registration Comparison between 2017, 2018 and 2019 (January to December)', size=12, color='dimgrey', pad=20)


plt.tight_layout()
plt.show()


# From the charts above we can conclude:
# 
# 1. Registration of new users has a ranging trend along the time. We can see some seasonality with peaks at specific months.
# 2. Saturdays experience the highest amount of registration closely followed by Friday and Tuesday, while Sunday experience the least amounts of registrations

# In[12]:


#chart Marketing Channel
fig = plt.figure(constrained_layout=True, figsize=(30, 15))
gs = GridSpec(2, 2, figure=fig)
ax1 = fig.add_subplot(gs[0, 0])

single_countplot(data, x='Marketing Channel', ax=ax1,  palette='YlGnBu')
ax1.set_title('Marketing Channel', size=12, color='dimgrey')

plt.show()


# In[13]:


pd.crosstab(data['Marketing Channel'], data['registration_year'], dropna=False)


# In[14]:


pd.crosstab(data['Marketing Channel'], data['registration_year'], dropna=False).plot(kind="bar", 
                                   figsize=(20,9), color=["skyblue","salmon","green"])


# from the charts above we can see that
# 1. About 30% of all new registered come through the SEO channel with peak in 2017. It experience a 43% decrease in 2019.
# 2. Club experienced a 3000% increase in 2019 compared to 2018

# In[15]:


fig = plt.figure(constrained_layout=True, figsize=(20, 15))
gs = GridSpec(2, 2, figure=fig)
ax1 = fig.add_subplot(gs[0, 0])

colors = ['darkslateblue', 'cornflowerblue', 'silver', 'darkviolet', 'crimson']
label_names = data['RegistrationDeviceType'].value_counts().index
donut_plot(data, col='RegistrationDeviceType', ax=ax1, label_names=label_names, colors=colors,
           title='Count of Customer Registration Device Type', text=f'{len(data)} \nCustomer \nRegistration')
plt.show()


# In[16]:


#chart of Customer Region 
fig = plt.figure(constrained_layout=True, figsize=(18, 12))
gs = GridSpec(2, 2, figure=fig)
ax1 = fig.add_subplot(gs[0, 0])

single_countplot(y='Region', ax=ax1, df=data, palette='viridis')
ax1.set_title('Customer Region', size=12, color='dimgrey')

plt.show()


# In[17]:


#Evolution of of Customer Registration by Region
pd.crosstab(data['Region'], data['registration_year'], dropna=False).plot(kind="bar", 
                                   figsize=(20,9), color=["skyblue","salmon","green"])


# In[ ]:





# In[ ]:





# ## 4. Data Preprocessing

# From the first experiment i will the filling the blank spaces in the categorical feature with "Absent" under the assumption the each count column of every category represents a day worth of data. By replacing with absent it means no user data was record that day due to the client being inactive. 

# In[18]:


#Dropping NaN by region 
df=data.copy()
df.dropna(subset=['Region'], inplace=True)
len(df)


# In[19]:


#replacing blank spaces with Absent
for index, value in df.dtypes.items(): 
    if value == 'object':
        df[index] = df[index].fillna('Absent')

df.info()


# ## 5. Modelling

# ### RandomForest 

# In[20]:


from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
for index, value in df.dtypes.items(): 
    if value == 'object':
        df[index] = labelencoder.fit_transform(df[index])
df.head(10)


# In[21]:


X = df.drop(['LTV','user ID','DateFormatted','RegistrationAppExtension','registration_month'],axis=1)
y = df['LTV']
# Splitting dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 42)


# In[22]:


# RandomForest 
def rfr(X_train, X_test, y_train, y_test):    

    model = RandomForestRegressor()

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
        
     # Metrics
    mae = np.mean(abs(predictions - y_test))
    rmse = mean_squared_error(y_test, predictions, squared=False)
    r2score = r2_score(y_test, predictions)
    
    print("MAE: ", mae)
    print("RMSE: ", rmse)
    print("R2: ",r2score)
        
    return model


# In[23]:


randomforestmodel=rfr(X_train, X_test, y_train, y_test)


# ### KNeighborsRegressor

# In[24]:


# KNeighborsRegressor
def KNr(X_train, X_test, y_train, y_test):    

    model = KNeighborsRegressor(5)

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
        
     # Metrics
    mae = np.mean(abs(predictions - y_test))
    rmse = mean_squared_error(y_test, predictions, squared=False)
    r2score = r2_score(y_test, predictions)
    
    print("MAE: ", mae)
    print("RMSE: ", rmse)
    print("R2: ",r2score)
        
    return model


# In[25]:


KNeighborsmodel=KNr(X_train, X_test, y_train, y_test)


# ### CatBoost

# In[26]:


def catboost_model(X_train, y_train,X_test, y_test):
    model = CatBoostRegressor(
        random_seed = 400,
        loss_function = 'RMSE',
        iterations=400,
    )
    
    model.fit(
        X_train, y_train,
        eval_set=(X_valid, y_valid),
        verbose=False
    )
    
    predictions = model.predict(X_test)
    mae = np.mean(abs(predictions - y_test))
    rmse = mean_squared_error(y_test, predictions, squared=False)
    r2score = r2_score(y_test, predictions)
    
    print("MAE: ", mae)
    print("RMSE: ", rmse)
    print("R2: ",r2score)
    
    return model


# In[27]:


X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2, 
                                                    random_state=52)


# In[28]:


catboostmodel=catboost_model(X_train, y_train,X_test, y_test)


# ### Feature Selection

# The performance of the models above isn't great, i will be carring out forward feature selection and re-train the models and compare the performances. i will by selecting 80 features and then gradually keep on reducing the amount and compare the performances

# In[29]:


# find and remove correlated features
def correlation(dataset, threshold):
    col_corr = set()  # Set of all the names of correlated columns
    corr_matrix = dataset.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > threshold: # we are interested in absolute coeff value
                colname = corr_matrix.columns[i]  # getting the name of column
                col_corr.add(colname)
    return col_corr

corr_features = correlation(X_train, 0.8)
print('correlated features: ', len(set(corr_features)) )


# In[30]:


corr_features


# In[31]:


# # removed correlated  features
# X_train.drop(labels=corr_features, axis=1, inplace=True)
# X_test.drop(labels=corr_features, axis=1, inplace=True)

# X_train.shape, X_test.shape


# #### 80 Features

# In[32]:


from sklearn.feature_selection import SequentialFeatureSelector
sfs = SequentialFeatureSelector(RandomForestRegressor(),
                                n_features_to_select=80,
                                scoring='r2',
                                cv=3
                               )
sfs.fit(X, y)
sfs.get_feature_names_out()


# In[33]:


X1=sfs.transform(X)


# In[34]:


X_train, X_test, y_train, y_test = train_test_split(X1, y, test_size=0.2, 
                                                    random_state=42)


# In[35]:


randomforestmodel1=rfr(X_train, X_test, y_train, y_test)


# In[36]:


KNeighborsmodel1=KNr(X_train, X_test, y_train, y_test)


# In[37]:


X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2, 
                                                    random_state=52)
catboostmodel=catboost_model(X_train, y_train,X_test, y_test)


# #### 50 Features

# In[38]:


from sklearn.feature_selection import SequentialFeatureSelector
sfs2 = SequentialFeatureSelector(RandomForestRegressor(),
                                n_features_to_select=50,
                                scoring='r2',
                                cv=3
                               )
sfs2.fit(X, y)
sfs2.get_feature_names_out()


# In[39]:


X2=sfs.transform(X)


# In[40]:


X_train, X_test, y_train, y_test = train_test_split(X1, y, test_size=0.2, 
                                                    random_state=42)


# In[41]:


randomforestmodel2=rfr(X_train, X_test, y_train, y_test)


# In[42]:


KNeighborsmodel2=KNr(X_train, X_test, y_train, y_test)


# In[43]:


X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2, 
                                                    random_state=52)
catboostmodel2=catboost_model(X_train, y_train,X_test, y_test)


# after using feature selection, the overall Performance of all the models reduced.

# ## Experiment 2

# In this experiment i will be dropping every column that is originally over 50% empty.

# In[44]:


#dropping unwanted columns and empty columns
pre_data=df.drop(['user ID','DateFormatted','registration_month',
                      'registration_year_month','registration_day',
                      '0_device','1_device','2_device','3_device',
                      '4_device','5_device','6_device','0_category',
                      '1_category','2_category','3_category','4_category',
                      '5_category','6_category','0_status','1_status',
                      '2_status','3_status','4_status','5_status','6_status'
                     ], axis=1)


# In[45]:


pre_data.shape


# In[46]:


X = pre_data.drop(['LTV'],axis=1)
y = pre_data['LTV']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state = 42)
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2, random_state=52)


# In[47]:


randomforestmodel2=rfr(X_train, X_test, y_train, y_test)


# In[48]:


KNeighborsmodel2=KNr(X_train, X_test, y_train, y_test)


# In[49]:


catboostmodel2=catboost_model(X_train, y_train,X_test, y_test)


# the model still isn't performing great

# ### Experiment 3

# i will merging all the count columns into a single total for all the count columns

# In[50]:


predata=pre_data.copy()
predata['total_stakes']=predata['0_total_stakes']+predata['1_total_stakes']+predata['2_total_stakes']+predata['3_total_stakes']+predata['4_total_stakes']+predata['5_total_stakes']+predata['6_total_stakes']
predata['count_stakes']=predata['0_count_stakes']+predata['1_count_stakes']+predata['2_count_stakes']+predata['3_count_stakes']+predata['4_count_stakes']+predata['5_count_stakes']+predata['6_count_stakes']
predata['bonus_stakes']=predata['0_bonus_stakes']+predata['1_bonus_stakes']+predata['2_bonus_stakes']+predata['3_bonus_stakes']+predata['4_bonus_stakes']+predata['5_bonus_stakes']+predata['6_bonus_stakes']
predata['cash_hold']=predata['0_cash_hold']+predata['1_cash_hold']+predata['2_cash_hold']+predata['3_cash_hold']+predata['4_cash_hold']+predata['5_cash_hold']+predata['6_cash_hold']
predata['net_gaming_revenue_y']=predata['0_net_gaming_revenue_y']+predata['1_net_gaming_revenue_y']+predata['2_net_gaming_revenue_y']+predata['3_net_gaming_revenue_y']+predata['4_net_gaming_revenue_y']+predata['5_net_gaming_revenue_y']+predata['6_net_gaming_revenue_y']
predata['total_freebets']=predata['0_total_freebets']+predata['1_total_freebets']+predata['2_total_freebets']+predata['3_total_freebets']+predata['4_total_freebets']+predata['5_total_freebets']+predata['6_total_freebets']
predata['count_deposits']=predata['0_count_deposits']+predata['1_count_deposits']+predata['2_count_deposits']+predata['3_count_deposits']+predata['4_count_deposits']+predata['5_count_deposits']+predata['6_count_deposits']
predata['total_deposits']=predata['0_total_deposits']+predata['1_total_deposits']+predata['2_total_deposits']+predata['3_total_deposits']+predata['4_total_deposits']+predata['5_total_deposits']+predata['6_total_deposits']
predata['count_withdrawals']=predata['0_count_withdrawals']+predata['1_count_withdrawals']+predata['2_count_withdrawals']+predata['3_count_withdrawals']+predata['4_count_withdrawals']+predata['5_count_withdrawals']+predata['6_count_withdrawals']
predata['total_withdrawals']=predata['0_total_withdrawals']+predata['1_total_withdrawals']+predata['2_total_withdrawals']+predata['3_total_withdrawals']+predata['4_total_withdrawals']+predata['5_total_withdrawals']+predata['6_total_withdrawals']


# In[51]:


predata=predata.drop(['0_total_stakes','1_total_stakes','2_total_stakes','3_total_stakes','4_total_stakes',
                      '5_total_stakes','6_total_stakes','0_count_stakes','1_count_stakes','2_count_stakes',
                      '3_count_stakes','4_count_stakes','5_count_stakes','6_count_stakes','0_bonus_stakes',
                      '1_bonus_stakes','2_bonus_stakes','3_bonus_stakes','4_bonus_stakes','5_bonus_stakes',
                      '6_bonus_stakes','0_cash_hold','1_cash_hold','2_cash_hold','3_cash_hold','4_cash_hold',
                      '5_cash_hold','6_cash_hold','0_net_gaming_revenue_y','1_net_gaming_revenue_y',
                      '2_net_gaming_revenue_y','3_net_gaming_revenue_y','4_net_gaming_revenue_y',
                      '5_net_gaming_revenue_y','6_net_gaming_revenue_y','0_total_freebets','1_total_freebets',
                      '2_total_freebets','3_total_freebets','4_total_freebets','5_total_freebets','6_total_freebets',
                      '0_count_deposits','1_count_deposits','2_count_deposits','3_count_deposits','4_count_deposits',
                      '5_count_deposits','6_count_deposits','0_total_deposits','1_total_deposits','2_total_deposits',
                      '3_total_deposits','4_total_deposits','5_total_deposits','6_total_deposits','0_count_withdrawals',
                      '1_count_withdrawals','2_count_withdrawals','3_count_withdrawals','4_count_withdrawals',
                      '5_count_withdrawals','6_count_withdrawals','0_total_withdrawals','1_total_withdrawals',
                      '2_total_withdrawals','3_total_withdrawals','4_total_withdrawals','5_total_withdrawals',
                      '6_total_withdrawals',
                     ], axis=1)


# In[52]:


predata.shape


# In[53]:


X = predata.drop(['LTV'],axis=1)
y = predata['LTV']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state = 42)
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2, random_state=52)


# In[54]:


randomforestmodel2=rfr(X_train, X_test, y_train, y_test)


# In[55]:


KNeighborsmodel2=KNr(X_train, X_test, y_train, y_test)


# In[56]:


catboostmodel2=catboost_model(X_train, y_train,X_test, y_test)


# the Model still isn't performing well

# ## 6. Hyperparamter tuning 

# i will be attempting to tune the model using and randomsearchcv and gridsearhcv model to see if that will help in increase the performance of the models.

# #### RandomizedSearchCV

# In[57]:


X = predata.drop(['LTV'],axis=1)
y = predata['LTV']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state = 42)
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2, random_state=52)


# In[58]:


from sklearn.model_selection import RandomizedSearchCV
# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
print(random_grid)


# In[59]:


# Use the random grid to search for best hyperparameters
# First create the base model to tune
rf = RandomForestRegressor()
# Random search of parameters, using 3 fold cross validation, 
# search across 100 different combinations, and use all available cores
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)
# Fit the random search model
rf_random.fit(X_train, y_train)


# In[61]:


rf_random.best_params_


# In[72]:


def evaluate(model, X_test, y_test):
    predictions = model.predict(X_test)
    mae = np.mean(abs(predictions - y_test))
    rmse = mean_squared_error(y_test, predictions, squared=False)
    r2score = r2_score(y_test, predictions)
    
    print("MAE: ", mae)
    print("RSME: ", rmse)
    print("R2: ",r2score)
    
    return r2score


# In[74]:


random=evaluate(rf_random,X_test, y_test)


# #### GridSearchCV

# To use Grid Search, we make another grid based on the best values provided by random search:

# In[75]:


from sklearn.model_selection import GridSearchCV
# Create the parameter grid based on the results of random search 
param_grid = {'n_estimators': [100,200,400,600,800,1000],
             'min_samples_split': [1,2,3],
             'min_samples_leaf': [3,4,5],
             'max_features': ['auto'],
             'max_depth': [40,50,60,70,80],
             'bootstrap': [False]
             }
# Create a based model
rf = RandomForestRegressor()
# Instantiate the grid search model
grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, 
                          cv = 3, n_jobs = -1, verbose = 2)


# In[76]:


grid_search.fit( X_train, y_train)


# In[77]:


best_grid = grid_search.best_estimator_
grid = evaluate(best_grid,X_test, y_test )


# the result after hyperparameter tuning is fairly the same, very minimal change.

# ### 7.Stacking Ensemble

# This involves combining the predictions from multiple machine learning models on the same dataset, like bagging and boosting.

# In[68]:


# compare machine learning models for regression
 
# get a stacking ensemble of models
def get_stacking():
	# define the base models
	level0 = list()
	level0.append(('RandomForest', RandomForestRegressor()))
	level0.append(('KNN', KNeighborsRegressor()))
	level0.append(('CatBoost', CatBoostRegressor()))
	# define meta learner model
	level1 = LinearRegression()
	# define the stacking ensemble
	model = StackingRegressor(estimators=level0, final_estimator=level1, cv=5)
	return model
 
# get a list of models to evaluate
def get_models():
	models = dict()
	models['RandomForest'] = RandomForestRegressor()
	models['KNN'] = KNeighborsRegressor()
	models['CatBoost'] = CatBoostRegressor()
	models['stacking'] = get_stacking()
	return models
 
# evaluate a given model using cross-validation
def evaluate_model(model, X, y):
	cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
	scores = cross_val_score(model, X, y, scoring='r2', cv=cv, n_jobs=-1, error_score='raise')
	return scores
 


# In[69]:


# get the models to evaluate
models = get_models()
# evaluate the models and store results
results, names = list(), list()
for name, model in models.items():
	scores = evaluate_model(model, X, y)
	results.append(scores)
	names.append(name)
	print('>%s %.3f (%.3f)' % (name, mean(scores), std(scores)))
# plot model performance for comparison
pyplot.boxplot(results, labels=names, showmeans=True)
pyplot.show()


# ### Deep Learning

# i will be attempting to implement deep Learning neural network and comparing the performance to the current models.
# 

# In[71]:


# df.to_csv('df.csv')
# pre_data.to_csv('pre_data.csv')
# predata.to_csv('predata.csv')


# In[ ]:




