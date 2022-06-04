#!/usr/bin/env python
# coding: utf-8

# Perform Clustering(Hierarchical, Kmeans & DBSCAN) for the crime data and identify the number of clusters formed and draw inferences.
# 
# Data Description:
# Murder -- Muder rates in different places of United States
# Assualt- Assualt rate in different places of United States
# UrbanPop - urban population in different places of United States
# Rape - Rape rate in different places of United States

# In[122]:


#HIERACHICAL CLUSTERING LIBRARIES

from sklearn.cluster import AgglomerativeClustering
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
import scipy.cluster.hierarchy as sch


# In[123]:


cm2=pd.read_csv("E:\DATA SCIENCE\LMS\ASSIGNMENT\MY ASSIGNMENT\CLUSTERING\crime_data.csv")


# In[124]:


cm=cm2.rename({'Unnamed: 0':'place'},axis=1)
cm


# In[125]:


CM=cm.iloc[:,1:5]


# In[126]:


CM


# In[127]:


#normalization function
def norm_func(i):
    x = (i-i.min())/(i.max()-i.min())
    return(x)


# In[128]:


#normalization data frame
norm = norm_func(CM)


# In[129]:


norm


# In[ ]:


#creating dendrogram
dendrogram=sch.dendrogram(sch.linkage(norm,method='single'))


# In[ ]:


#creating clusters
hc = AgglomerativeClustering(n_clusters=3,affinity='euclidean',linkage='single')


# In[ ]:


hc


# In[ ]:


#Save clusers for chart
y_hc = hc.fit_predict(norm)


# In[ ]:


y_hc


# In[ ]:


#converting numpy array in pandas series
mc=pd.Series(y_hc)


# In[ ]:


mc


# In[ ]:


#create a dataframe
cl=pd.DataFrame(y_hc,columns=['clusters'])


# In[ ]:


pd.concat([cm,cl],axis=1)


# In[ ]:


# Perform K-Means clusering
#importing libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
import numpy as np
from sklearn.preprocessing import StandardScaler


# In[ ]:


# we already perform Normalizaion so now make a model
modelc=KMeans(n_clusters=3).fit(norm)


# In[ ]:


#getting labels of clusters assigned to each row
modelc.labels_


# In[ ]:


#converting numpy array in pandas series
mc=pd.Series(modelc.labels_)


# In[ ]:


#creating new column assign it to new column
cm['cust']=mc


# In[ ]:


cm


# In[195]:


cm.groupby(cm.cust).mean()


# #As we can see from above result, average Murder is highest belongs to cluster 1.
# average Assault is highest belongs to cluster 0.
# average Urbanpop is highest belongs to cluster 2.
# average Rape is highest belongs to cluster 0.
# 

# In[ ]:


#load libray of DBSCAN
from sklearn.cluster import DBSCAN
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[ ]:


#as we already performed NOrmalizaton in variable norm
#assume eps=0.8 and as columns is 4 minsample=5
dbscan = DBSCAN(eps=0.8,min_samples=5)
dbscan.fit(norm)


# In[ ]:


dbscan.labels_


# In[ ]:


#converting numpy array in pandas series
mcDB=pd.Series(dbscan.labels_)


# In[ ]:


#make a Dataframe and assigned new column for each row
dbscan3=pd.DataFrame(mcDB,columns=['clust'])


# In[ ]:


Checkclust=pd.concat([cm.iloc[:,0:4],dbscan3],axis=1)


# In[173]:


Checkclust


# #Q-2 Perform clustering (hierarchical,K means clustering and DBSCAN) for the airlines data to obtain optimum number of clusters. 
# Draw the inferences from the clusters obtained.
# 
# Data Description:
#  The file EastWestAirlines contains information on passengers who belong to an airline’s frequent flier program. For each passenger the data include information on their mileage history and on different ways they accrued or spent miles in the last year. The goal is to try to identify clusters of passengers that have similar characteristics for the purpose of targeting different segments for different types of mileage offers
# 
# ID --Unique ID
# 
# Balance--Number of miles eligible for award travel
# 
# Qual_mile--Number of miles counted as qualifying for Topflight status
# 
# cc1_miles -- Number of miles earned with freq. flyer credit card in the past 12 months:
# cc2_miles -- Number of miles earned with Rewards credit card in the past 12 months:
# cc3_miles -- Number of miles earned with Small Business credit card in the past 12 months:
# 
# 1 = under 5,000
# 2 = 5,000 - 10,000
# 3 = 10,001 - 25,000
# 4 = 25,001 - 50,000
# 5 = over 50,000
# 
# Bonus_miles--Number of miles earned from non-flight bonus transactions in the past 12 months
# 
# Bonus_trans--Number of non-flight bonus transactions in the past 12 months
# 
# Flight_miles_12mo--Number of flight miles in the past 12 months
# 
# Flight_trans_12--Number of flight transactions in the past 12 months
# 
# Days_since_enrolled--Number of days since enrolled in flier program
# 
# Award--whether that person had award flight (free flight) or not

# In[ ]:


#HIERACHICAL CLUSTERING LIBRARIES

from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
import scipy.cluster.hierarchy as sch


# In[ ]:


ew=pd.read_csv("E:\DATA SCIENCE\LMS\ASSIGNMENT\MY ASSIGNMENT\CLUSTERING\EastWestAirlines.csv")


# In[ ]:


ew


# In[206]:


ew1=ew.iloc[:,1:12]
ew1


# In[ ]:


def nomr_func(i):
    x=(i-i.min())/(i.max()-i.min())
    return (x)
norm5=norm_func(ew1)


# In[ ]:


norm5


# In[ ]:


hc1=AgglomerativeClustering(n_clusters=4,affinity='euclidean',linkage='single')


# In[192]:


#creating dendrogram
dendrogram = sch.dendrogram(sch.linkage(norm5,method='single'))


# In[ ]:


y_hc1=hc1.fit_predict(norm5)


# In[181]:


y_hc1


# In[182]:


clusters1=pd.DataFrame(y_hc1,columns=['cluster'])


# In[183]:


pd.concat([ew,clusters1],axis=1)


# In[184]:


# Perform K-Means clusering
#importing libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
import numpy as np
from sklearn.preprocessing import StandardScaler


# In[185]:


# As we already load file and normalizaton function.Perform KMeans clustering model
model_e=KMeans(n_clusters=5).fit(norm5)


# In[186]:


#getting labels of clusters assigned to each row
model_e.labels_


# In[187]:


#convering numpy array into pandas series object
md3=pd.Series(model_e.labels_)


# In[188]:


md3


# In[189]:


#creating new column and assign it to each row
ew['clusterKmeans']=md3


# In[190]:


ew.iloc[0:14]


# In[200]:


ew.iloc[:,1:12].groupby(ew.clusterKmeans).mean()


# In[202]:


#DBscan load library
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[203]:


#we already load data so make array 
array3=ew1.values


# In[210]:


stscaler=StandardScaler().fit(array3)
x= stscaler.transform(array3)


# In[208]:


# now perform DBSCAN
dbscan5=DBSCAN(eps=0.8,min_samples=13)
dbscan5.fit(x)


# In[211]:


dbscan5.labels_


# In[212]:


ew6=pd.DataFrame(dbscan5.labels_,columns=['clusersDBSCAN'])
ew6


# In[213]:


pd.concat([ew1,ew6],axis=1)

