#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


df=pd.read_excel("C:/Users/alekh/Downloads/Data_file.xlsx")
df.head()


# In[3]:


#let us find missing values
df.nunique()


# In[5]:


df.shape


# In[6]:


df.isnull().sum()


# In[7]:


#pre-processing
college_dummies=pd.get_dummies(df['College'])
college_dummies.drop('Tier 3',axis=1,inplace=True)
college_dummies.head()


# In[9]:


role_dummies=pd.get_dummies(df['Role'])
role_dummies.drop('Manager',axis=1,inplace=True)
role_dummies.head()


# In[11]:


city_dummies=pd.get_dummies(df['City type'])
city_dummies.drop('Metro',axis=1,inplace=True)
city_dummies.head()


# In[12]:


df.drop(['S.No.','College','Role','City type'],axis=1,inplace=True)
new_df=pd.concat([college_dummies,role_dummies,city_dummies,df],axis=1)
new_df.head()


# In[23]:


new_df.head()


# In[33]:


new_df.drop('Executive',axis=1,inplace=True)


# In[34]:


new_df.head()


# In[37]:


from sklearn import preprocessing
value=new_df.values
min_max_scalar=preprocessing.MinMaxScaler()
val_scaled=min_max_scalar.fit_transform(value)
normalize_df=pd.DataFrame(val_scaled)
normalize_df.columns=['College_T1','College_T2','City_non metro','previous job changes','Exp (Months)']
df.rename(columns={'CTC':'Actual CTC'},inplace=True)


# In[38]:


#Training of model
X=normalize_df
Y=df['Actual CTC']


# In[39]:


from sklearn.linear_model import LinearRegression
model=LinearRegression()


# In[40]:


model.fit(X,Y)


# In[52]:


#Testing
test_df=pd.read_excel("C:/Users/alekh/Downloads/Test_data_file.xlsx")
test_df.head()


# In[46]:


test_df.nunique()
test_df.head()


# In[48]:


test_df.shape


# In[53]:


test_df.isnull().sum()


# In[54]:


test_df=test_df.drop(['College','Role','City type','Predicted CTC'],axis='columns')
test_df.head()


# In[60]:


new_test_df=test_df.copy(deep=True)
new_test_df.drop('Actual CTC',axis=1,inplace=True)
new_test_df.drop(['Role_Manager','Graduation marks','previous CTC'],axis=1,inplace=True)
new_test_df.head()


# In[61]:


#Normalize the data
test_val=new_test_df.values
test_val_scaled=min_max_scaler.fit_transform(test_val)
test_normalize_df=pd.DataFrame(test_val_scaled)
test_normalize_df.columns=['College_T1','College_T2','City_non metro','previous job changes','Exp (Months)']
test_normalize_df.head()


# In[63]:


#evaluate the modal
X_test=test_normalize_df
Y_test=test_df['Actual CTC']
predict=model.predict(X_test)
prediction=predict.reshape(-1,1)
print(prediction)


# In[65]:


test_df['Actual CTC']


# In[69]:


#to calculate mean squared error to evalaute
from sklearn.metrics import mean_squared_error
print('MSE',mean_squared_error(Y_test,prediction))


# In[ ]:





# In[ ]:




