
# coding: utf-8

# ## This is an example of categorical data analysis
# ## When digging into features in the parser, conclude with 
# 
# * variable: variable name 
# * type: identification of the type of the vairable -- numerical/categorical
# * segment: general, network, security, ... etc
# * expectation: our expectations about the variable influence on the predict problems, 'high'/'medium'/'low'
# * conclusion: our conclusion about the importance of the variable, after we analyze the data, the same categorial scale with expectation
# * comments: any general comments that occured to us

# In[1]:


import pandas as pd


# In[2]:


import sys
sys.path.insert(0, "/home/qialiu/work/lib//python2.7/site-packages")


# In[3]:


import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')


# In[4]:


import seaborn as sns
sns.set(color_codes=True)


# In[5]:


rpms = spark.read.parquet("/user/csams/insights_data/parquet/2017-11-07/warehouse/insights_parsers_installed_rpms_installedrpms")


# In[6]:


rpms.columns


# In[8]:


rpms.registerTempTable("rpms")


# In[9]:


sql = rpms.sql_ctx.sql


# ## task 1: group by account
# ## check the number of rpm installed per account, sorted in a descending order 

# In[11]:


sql("select account, count(distinct name) as distinct_rpms_installed from rpms group by account order by 2 desc").show()


# In[12]:


distinct_rpms_count_acc = sql("select account, count(distinct name) as distinct_rpms_installed from rpms group by account order by 2 desc")


# In[13]:


distinct_rpms_count_acc_df = distinct_rpms_count_acc.toPandas()


# In[15]:


discint_rpms_cnt = distinct_rpms_count_acc_df['distinct_rpms_installed'].as_matrix()


# In[17]:


from scipy import stats, integrate


# In[19]:


sns.distplot(discint_rpms_cnt)


# In[20]:


from scipy.stats import norm
sns.distplot(discint_rpms_cnt, fit=norm)
fig = plt.figure() 
res = stats.probplot(discint_rpms_cnt, plot=plt)


# In[21]:


log_discint_rpms_cnt = np.log(discint_rpms_cnt)


# In[22]:


sns.distplot(log_discint_rpms_cnt, fit=norm)
fig = plt.figure() 
res = stats.probplot(log_discint_rpms_cnt, plot=plt)


# ## task 2: univariate: the distribution of rpm_name only

# In[23]:


rpms.select('name').show()


# In[24]:


t1 = rpms.sample(False, 0.01, 42)


# In[26]:


rpms_pd = t1.toPandas()


# ### For each rpm name, counting # of accounts has the rpm installed

# In[28]:


rpms_pd.groupby('name').agg('count')['account'].plot(kind = 'pie', figsize = (20, 20))


# ## top 50 most frequently installed rpms per account

# In[29]:


rpms_pd.groupby('name').agg('count')['account'].sort_values(ascending=False).head(50).plot(kind = 'barh', figsize = (20, 20))

