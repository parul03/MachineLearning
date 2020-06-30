
import pandas as pd # Import Pandas for data manipulation using dataframes
import numpy as np # Import Numpy for data statistical analysis 
import matplotlib.pyplot as plt # Import matplotlib for data visualisation
import random
import seaborn as sns
from fbprophet import Prophet


# In[2]:


# dataframes creation for both training and testing datasets 
avocado_df =pd.read_csv('avocado.csv')


# 
# - Date: The date of the observation
# - AveragePrice: the average price of a single avocado
# - type: conventional or organic
# - year: the year
# - Region: the city or region of the observation
# - Total Volume: Total number of avocados sold
# - 4046: Total number of avocados with PLU 4046 sold
# - 4225: Total number of avocados with PLU 4225 sold
# - 4770: Total number of avocados with PLU 4770 sold

# In[3]:


# Let's view the head of the training dataset
avocado_df.head()


# In[4]:


# Let's view the last elements in the training dataset
avocado_df.describe()


# In[5]:


avocado_df.isnull().sum()


# In[6]:


avocado_df.tail()


# In[ ]:





# # TASK #3: EXPLORE DATASET  

# In[7]:


avocado_df = avocado_df.sort_values('Date')


# In[8]:


# Plot date and average price
plt.figure(figsize =(10,10))
plt.plot(avocado_df['Date'], avocado_df['AveragePrice'])


# In[9]:


# Plot distribution of the average price
plt.figure(figsize =(10,10))
sns.distplot(avocado_df['AveragePrice'], color='b')


# In[10]:


sns.violinplot(y='AveragePrice', x='type',data=avocado_df)


# In[11]:


sns.set(font_scale=.7)
plt.figure(figsize=[25,12])
sns.countplot(x='region', data =avocado_df)
plt.xticks(rota# Plot a violin plot of the average price vs. avocado type
tion=45)


# In[13]:


# Bar Chart to indicate the number of regions 

sns.set(font_scale=0.7) 
plt.figure(figsize=[25,12])
sns.countplot(x = 'region', data = avocado_df)
plt.xticks(rotation = 45)


# In[12]:


# Bar Chart to indicate the count in every year
sns.set(font_scale=1.5) 
plt.figure(figsize=[25,12])
sns.countplot(x = 'year', data = avocado_df)
plt.xticks(rotation = 45)


# In[16]:


# plot the avocado prices vs. regions for conventional avocados
conventional =sns.catplot('AveragePrice','region',data=avocado_df[avocado_df['type']=='conventional'],hue='year',
                           height=20)


# In[19]:


# plot the avocado prices vs. regions for organic avocados
conventional =sns.catplot('AveragePrice','region',data=avocado_df[avocado_df['type']=='conventional'],hue='year',
                          height=20)


# # TASK 4: PREPARE THE DATA BEFORE APPLYING FACEBOOK PROPHET TOOL 

# In[20]:


avocado_prophet_df =avocado_df[['Date','AveragePrice']]


# In[21]:


avocado_prophet_df


# In[23]:


avocado_prophet_df =avocado_prophet_df.rename(columns={'Date':'ds' , 'AveragePrice': 'y'})


# In[24]:


avocado_prophet_df


# # TASK 5: UNDERSTAND INTUITION BEHIND FACEBOOK PROPHET

# ![image.png](attachment:image.png)

# ![image.png](attachment:image.png)

# ![image.png](attachment:image.png)

# # TASK 6: DEVELOP MODEL AND MAKE PREDICTIONS - PART A

# In[26]:


m= Prophet()
m.fit(avocado_prophet_df)


# In[28]:


# Forcasting into the future
future = m.make_future_dataframe(periods =365)
forecast =m.predict(future)


# In[29]:


forecast


# In[30]:


figure = m.plot(forecast, xlabel='Date', ylabel='Price')


# In[31]:


fig2= m.plot_components(forecast)


# # TASK 7: DEVELOP MODEL AND MAKE PREDICTIONS (REGION SPECIFIC) - PART B

# In[ ]:


# dataframes creation for both training and testing datasets 
avocado_df = pd.read_csv('avocado.csv')


# In[32]:


# Select specific region
avocado_df_sample =avocado_df[avocado_df['region']=='West']


# In[34]:


avocado_df_sample=avocado_df_sample.sort_values('Date')


# In[ ]:


plt.plot()


# In[36]:


avocado_df_sample=avocado_df_sample.rename(columns={'Date':'ds','AveragePrice':'y'})


# In[ ]:





# In[ ]:





# In[ ]:





# In[37]:


m = Prophet()
m.fit(avocado_df_sample)
# Forcasting into the future
future = m.make_future_dataframe(periods=365)
forecast = m.predict(future)


# In[38]:


figure = m.plot(forecast, xlabel='Date', ylabel='Price')


# In[39]:


figure3 = m.plot_components(forecast)



