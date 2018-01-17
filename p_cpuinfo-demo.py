
# coding: utf-8

# ## This is an example of numerical data analysis
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
import sys
sys.path.insert(0, "/home/qialiu/work/lib//python2.7/site-packages")
import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
import seaborn as sns
sns.set(color_codes=True)


# In[2]:


cpuinfo = spark.read.parquet("/user/csams/insights_data/parquet/2017-11-12/warehouse/insights_parsers_cpuinfo_cpuinfo")


# In[3]:


cpuinfo.columns


# In[5]:


cpuinfo.describe("cpu_count").show()


# In[6]:


cpuinfo.describe("cache_size").show()


# In[7]:


cpuinfo.describe("cpu_speed").show()


# In[8]:


cpuinfo.describe("socket_count").show()


# In[9]:


cpuinfo.describe("model_name").show()


# In[10]:


pd_cpuinfo = cpuinfo.toPandas()


# In[12]:


pd_cpuinfo.info()


# In[13]:


cpu_core = pd_cpuinfo[['cpu_speed', 'cache_size', 'cpu_count', 'socket_count']]


# In[14]:


from scipy import stats, integrate
from scipy.stats import norm
sns.distplot(cpu_core['cpu_speed'], fit=norm)
fig = plt.figure() 
res = stats.probplot(cpu_core['cpu_speed'], plot=plt)


# In[15]:


from scipy import stats, integrate
from scipy.stats import norm
sns.distplot(cpu_core['cache_size'], fit=norm)
fig = plt.figure() 
res = stats.probplot(cpu_core['cache_size'], plot=plt)


# In[16]:


from scipy import stats, integrate
from scipy.stats import norm
sns.distplot(cpu_core['cpu_count'], fit=norm)
fig = plt.figure() 
res = stats.probplot(cpu_core['cpu_count'], plot=plt)


# In[17]:


log_cpu_count = np.log(cpu_core['cpu_count'])


# In[18]:


sns.distplot(log_cpu_count, fit=norm)
fig = plt.figure() 
res = stats.probplot(log_cpu_count, plot=plt)


# In[19]:


from scipy import stats, integrate
from scipy.stats import norm
sns.distplot(cpu_core['socket_count'], fit=norm)
fig = plt.figure() 
res = stats.probplot(cpu_core['socket_count'], plot=plt)


# In[20]:


log_socket_count = np.log(cpu_core['socket_count'])


# In[22]:


from pandas.tools.plotting import scatter_matrix


# In[24]:


attributes = ["cpu_speed", "cache_size", "cpu_count", "socket_count"]


# In[38]:


cpu_core_T = cpu_core.transpose()


# ### Calculate correlation coefficient

# In[41]:


from numpy import corrcoef
print attributes
corrcoef(cpu_core_T)


# In[25]:


scatter_matrix(cpu_core[attributes], figsize=(12,8))

