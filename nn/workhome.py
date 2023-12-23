# -*- coding: utf-8 -*-
"""
Created on Sat Dec 16 20:07:47 2023

@author: GoogleTech
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# استيراد البيانات
data = pd.read_csv('economic_data.csv')
print(data)
print(data.describe())
print(data.head())

# استخراج المتغيرات
x=data.iloc[:,:1]  
y=data.iloc[:,1]

print(x)
print(y)
# بناء نموذج الانحدار الخطي
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(x,y )

# استخراج معامل الانحدار ونقطة التقاطع
print(model.coef_)
print(model.intercept_) 

# رسم مخطط تشتت للبيانات والخط المستقيم
plt.scatter(x, y)
plt.plot(x, model.predict(x), color='red')
plt.show()

# تنبؤ بإيرادات السنة 
print(model.predict([[2030]]))
print(model.predict([[2040]]))
print(model.predict([[2050]]))

model.score(x,y)

