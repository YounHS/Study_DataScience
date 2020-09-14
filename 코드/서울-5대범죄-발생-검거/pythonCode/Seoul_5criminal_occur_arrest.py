
# coding: utf-8

# In[2]:


import pandas as pd


# In[ ]:


# 데이터셋 경로, 파일명 설정 및 csv 파일 읽기

# dataset_dir = './#dataset/5criminal_occur_arr/'
# filename = dataset_dir + '2001'
# df = pd.read_csv(filename + '.csv', encoding='euc-kr')


# In[ ]:


# 데이터셋에 date 컬럼 추가 및 날짜형식으로 변경

# df['date'] = '2001'
# df['date'] = pd.to_datetime(df['date'])


# In[ ]:


# 정제한 데이터셋 재저장

# df.to_csv(filename+ '.csv', index=False)


# In[6]:


dataset_dir = '../dataset/'
df2000 = dataset_dir + '2000'
df2001 = dataset_dir + '2001'
df2002 = dataset_dir + '2002'
df2003 = dataset_dir + '2003'
df2004 = dataset_dir + '2004'
df2005 = dataset_dir + '2005'
df2006 = dataset_dir + '2006'
df2007 = dataset_dir + '2007'
df2008 = dataset_dir + '2008'
df2009 = dataset_dir + '2009'
df2010 = dataset_dir + '2010'
df2011 = dataset_dir + '2011'
df2012 = dataset_dir + '2012'
df2013 = dataset_dir + '2013'
df2014 = dataset_dir + '2014'
df2015 = dataset_dir + '2015'
df2016 = dataset_dir + '2016'
df2017 = dataset_dir + '2017'
df2018 = dataset_dir + '2018'


# In[77]:


# df = pd.read_csv(df2018 + '.csv', encoding='euc-kr')


# In[78]:


# df['date'] = '2018'
# df['date'] = pd.to_datetime(df['date'])


# In[79]:


# df.to_csv(df2018 + '.csv', index=False)


# In[80]:


# df


# In[7]:

csv2000 = pd.read_csv(df2000 + '.csv')
csv2001 = pd.read_csv(df2001 + '.csv')
csv2002 = pd.read_csv(df2002 + '.csv')
csv2003 = pd.read_csv(df2003 + '.csv')
csv2004 = pd.read_csv(df2004 + '.csv')
csv2005 = pd.read_csv(df2005 + '.csv')
csv2006 = pd.read_csv(df2006 + '.csv')
csv2007 = pd.read_csv(df2007 + '.csv')
csv2008 = pd.read_csv(df2008 + '.csv')
csv2009 = pd.read_csv(df2009 + '.csv')
csv2010 = pd.read_csv(df2010 + '.csv')
csv2011 = pd.read_csv(df2011 + '.csv')
csv2012 = pd.read_csv(df2012 + '.csv')
csv2013 = pd.read_csv(df2013 + '.csv')
csv2014 = pd.read_csv(df2014 + '.csv')
csv2015 = pd.read_csv(df2015 + '.csv')
csv2016 = pd.read_csv(df2016 + '.csv')
csv2017 = pd.read_csv(df2017 + '.csv')
csv2018 = pd.read_csv(df2018 + '.csv')


# In[9]:


csv2012.columns.values[3] = "건수"
csv2012


# In[10]:


yearInter_df = pd.concat([csv2000, csv2001, csv2002, csv2003, csv2004, csv2005, csv2006, csv2007, csv2008, csv2009, csv2010, csv2011, csv2012, csv2013, csv2014, csv2015, csv2016, csv2017, csv2018], ignore_index=True, axis = 0)


# In[11]:


yearInter_df = yearInter_df.dropna(axis=0)


# In[12]:


yearInter_df = yearInter_df.reset_index()


# In[17]:


# del yearInter_df['index']
# del yearInter_df['level_0']
yearInter_df.to_csv('yearInter.csv')

