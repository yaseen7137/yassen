#!/usr/bin/env python
# coding: utf-8

# بدايةً نقوم بتضمين المكتبات الأساسيَّة كمكتبة التَّعامل مع المصفوفات نام باي ومكتبة التعامل مع الملفَّات بانداز ومكتبة الرَّسم البياني ماتبلوت ومكتبة اس كي للتَّعلُّم لبناء النموذج

# In[11]:


import pandas as pd
import matplotlib.pyplot as plt
import sklearn as sk


#  raed_csv  نقوم بقراءة  ملف قاعدة البيانات آيريس باستخدام تابع القراءة

# In[13]:


data = pd.read_csv("iris.csv")


# نقوم بعزل البيانات (الأعمدة الأربعة الأولى) عن عمود القرار (العمود الخامس) لتكون بيانات دخلٍ لعمليَّة التَّدريب

# In[14]:


x = data.iloc[:,:4]


# نقوم بعزل عمود القرار ليتم مقارنته مع نتيجة التَّدريب

# In[15]:



y = data.iloc[:,4:]


#  لتقسيم البيانات بنسبة ثمانون بالمئة للتدريب وعشرون بالمئة للاختبار train_test_split نستخدم المكتبة اس كي للتَّعلُّم ونستدعي التابع

# In[16]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2,random_state = 42)


#  من مكتبة اس كي للتَّعلُّم قسم الشَّجرة وإنشاء كائن منها وسيحتوي على توابع وعمليَّات التَّصنيف DecisionTreeClaSsifier نقوم ببناء قالب النّموذج من خلال تضمين النَّموذج

# In[17]:


from sklearn.tree import DecisionTreeClassifier
dec_tree = DecisionTreeClassifier()


# لملاءمة النَّموذج fit نستخدم التَّابع  

# In[18]:


dec_tree.fit(x_train,y_train)


#    لحساب دقِّة النَّموذج ومن ثمَّ نقوم بإظهارها score نستخدم التابع    

# In[19]:



print("Decision Tree Classification Score: ",dec_tree.score(x_test,y_test))


#  لرسم الشّجرة النِّهائيَّة للنَّموذج وثمّ نقوم بإظهارها plot_tree نستخدم التابع

# In[10]:


sk.tree.plot_tree(dec_tree)
plt.show()


# In[ ]:




