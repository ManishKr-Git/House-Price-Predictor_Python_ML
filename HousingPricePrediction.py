#!/usr/bin/env python
# coding: utf-8

# ## Housing Price Predictor

# In[1]:


import pandas as pd


# In[2]:


df = pd.read_csv("data.csv")


# In[3]:


df.head()


# In[4]:


df.info()


# In[5]:


df.describe()


# ## For ploting Histogram

# In[6]:


# %matplotlib inline
# import matplotlib.pyplot as plt
# df.hist(bins=50,figsize = (20,15))


# ## USer defines train_test_split function

# In[7]:


# import numpy as np
# def split_train_test(data,test_ratio):
#     np.random.seed(42)
#     shuffled = np.random.permutation(len(data))
#     test_set_size = int(len(data)*test_ratio)
#     test_indeces = shuffled[:test_set_size]
#     train_indeces = shuffled[test_set_size:]
#     return data.iloc[train_indeces],data.iloc[test_indeces]
        


# In[8]:


from sklearn.model_selection import train_test_split
train_set,test_set = train_test_split(df,test_size = 0.2,random_state = 42)


# In[9]:


print(f"rows in train_set: {len(train_set)}\nrows in train_set: {len(test_set)}\n")


# In[10]:


from sklearn.model_selection import StratifiedShuffleSplit
split = StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=42)
for train_indeces,test_indeces in split.split(df,df['CHAS']):
    strat_train_set = df.loc[train_indeces]
    strat_test_set = df.loc[test_indeces]


# In[11]:


print(f"rows in train_set: {len(strat_train_set)}\nrows in train_set: {len(strat_test_set)}\n")


# In[12]:


df =strat_train_set.copy()


# ## Looking for Correlations

# In[13]:


corr_matrix = df.corr()
corr_matrix['MEDV'].sort_values(ascending = False)


# In[14]:


# from pandas.plotting import scatter_matrix
# attr = ['MEDV','RM','ZN','LSTAT']
# scatter_matrix(df[attr],figsize = (12,8))


# In[15]:


df.plot(kind='scatter',x = 'RM',y='MEDV',alpha = 0.5)


# ## Trying out attribute combinations

# In[16]:


df['TAXPRM'] = df['TAX']/df['RM']


# In[17]:


corr_matrix = df.corr()
corr_matrix['MEDV'].sort_values(ascending = False)


# In[18]:


df.plot(kind='scatter',x = 'TAXPRM',y='MEDV',alpha = 0.5)


# In[19]:


df.shape


# In[20]:


df.describe()


# In[21]:


df =strat_train_set.drop('MEDV',axis = 1)
df_labels = strat_train_set['MEDV'].copy()


# ## Handling Missing Column

# In[22]:


from sklearn.impute import SimpleImputer
imputer  = SimpleImputer(strategy = 'median')
imputer.fit(df)


# In[23]:


X = imputer.transform(df)
df_tr = pd.DataFrame(X,columns=df.columns)
df_tr.describe()


# ## CREATING PIPELINE

# In[24]:


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
my_pipeline = Pipeline([('imputer',SimpleImputer(strategy = 'median')),
                       ('std_scaler',StandardScaler())])


# In[25]:


df_tr = my_pipeline.fit_transform(df)


# In[26]:


df_tr


# In[27]:


df_tr.shape


# ## Selecting desired model

# In[35]:


# from sklearn.linear_model import LinearRegression
# model = LinearRegression()
# from sklearn.tree import DecisionTreeRegressor
# model = DecisionTreeRegressor()
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor()
model.fit(df_tr,df_labels)


# In[36]:


some_data = df.iloc[:5]
some_labels = df_labels.iloc[:5]
prepared_data = my_pipeline.transform(some_data)
model.predict(prepared_data)


# In[37]:


list(some_labels)


# ## Evaluating the model

# In[38]:


import numpy as np
from sklearn.metrics import mean_squared_error
housing_prediction = model.predict(df_tr)
lin_mse = mean_squared_error(df_labels,housing_prediction)
lin_rmse  = np.sqrt(lin_mse)


# In[39]:


lin_mse


# ## Cross Validation

# In[40]:


from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, df_tr, df_labels, scoring="neg_mean_squared_error", cv=10)
rmse_score = np.sqrt(-scores)


# In[41]:


rmse_score


# In[42]:


#Hence best result is given by RnadomForestRegressor


# ## Saving the model

# In[43]:


from joblib import dump,load
dump(model,'Predictor.joblib')


# In[46]:


x_test = strat_test_set.drop('MEDV',axis = 1)
y_test = strat_test_set['MEDV'].copy()
x_prepared = my_pipeline.transform(x_test)
final_prediction = model.predict(x_prepared)
final_mse = mean_squared_error(y_test,final_prediction)
final_rmse = np.sqrt(final_mse)


# In[47]:


final_rmse


# In[48]:


print(final_prediction,list(y_test))


# In[49]:


prepared_data[0]


# In[ ]:




