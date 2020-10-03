# UAS-Kecerdasan-Buatan
ini adalah hasil untuk memenuhi syarat tugas UAS Kecerdasan Buatan
#membuat import di library
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression

df = pd.DataFrame([[3,80],[2,65],[4,90],[3,81],[4,97],[3,81],[1,50],[3,86],[5,95],
[3,84],[3,84],[5,92],[3,86],[4,92],[3,85],[2,70],[2,55],[4,92],[3,88],[3,89],[4,85],
[4,85],[1,50],[2,60],[2,62],[3,74],[3,79],[4,84],[3,87],[2,68]])
df.columns = ['y', 'x']

x_train = df['x'].values[:, np.newaxis] 
y_train = df['y'].values

lm = LinearRegression()
lm.fit(x_train, y_train)

print('Koefesien = ' + str(lm.coef_))
print('Intercept = ' + str(lm.intercept_))

x_test = [[0],[5]]
p = lm.predict(x_test)
print(p)
print(' ')

pb = lm.predict(x_train)
dfc = pd.DataFrame({'x': df['x'],'y': pb})
plt.scatter(df['x'],df['y'])
plt.plot (dfc['x'], dfc['y'], color = 'green')
plt.xlabel ('NILAI')
plt.ylabel ('BELAJAR')
plt.title ('DIAGRAM PENGARUH BELAJAR TERHADAP NILAI DALAM MATA PELAJARAN BAHASA PEMOGRAMAN JAVA')
print('Mutadin / 171011400031')
plt.show()
