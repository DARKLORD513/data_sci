import pandas as pd
import matplotlib.pyplot as plt
import math
import random
import numpy as np
from sklearn.linear_model import LinearRegression

data = pd.read_csv('games.csv')
meta1 = data['meta_score']
user1 = data['user_review']

#using min max formula
meta = meta1-0/100-0
user = user1-0/10-0

def meta_stats(data):
    mean = data.mean()
    median = data.median()
    var = data.var()
    std = data.std()
    occur = data.count()
    mode = data.mode().iloc[0]

    print(f"Statistics for Meta score are as folows :")
    print(f"Mean : {mean}")
    print(f"Median : {median}")
    print(f"Mode : {mode}")
    print(f"Occurences : {occur}")
    print(f"Variance : {var}")
    print(f"Standard Deviation : {std}")
meta_stats(data=meta1)

def pearson():
    summation = sum(ai * bi for ai, bi in zip(meta, user))
    mean_a = meta.mean()
    mean_b = user.mean()
    n = 17435
    part1 = n*mean_a*mean_b
    numerator = summation - part1
    std_a = meta.std()
    std_b = user.std()
    deno = (n-1)*std_a*std_b
    print("\n \nThe co-efficient is :", numerator/deno)
pearson()

def normal_meta():
    mean = meta1.mean()
    std = meta1.std()
    x = meta1.sample().values[0]
    pi = 3.14
    e = 2.71
    power = -1/2*((x - mean)/std)**2
    num = math.exp(e**power)
    sq = math.sqrt(2*pi)
    deno = std*sq
    print(f"Normal distribution value for meta score {x} is: {num/deno}")
normal_meta()

#! 1d histogram
plt.hist(meta1, bins = 15, edgecolor = 'black')
plt.xlabel('Score out of 100')
plt.ylabel('Number of games')
plt.title('Meta Score')
plt.show()

#! 2d Histogram
plt.hist2d(user1,meta1,bins=[16,6],cmap='Blues')
plt.xlabel('User Review')
plt.ylabel('Meta Review')
plt.title('User Review vs. Meta Review')
plt.colorbar()
plt.show()

#! Scatterplot
plt.scatter(user1, meta1, label='Data Points', color='blue', marker='o',s=2)
plt.xlabel('User Review')
plt.ylabel('Meta Review')
plt.title('Scatterplot')
plt.legend()
plt.show()

#! Regression line
plt.scatter(user1, meta1, label='Data Points', color='blue', marker='o',s=2)
model = LinearRegression()
X = user1.values.reshape(-1, 1)
y = meta1.values.reshape(-1, 1)
model.fit(X, y)

slope = model.coef_[0][0]
intercept = model.intercept_[0]
predicted_review = model.predict(X)

plt.plot(user1, predicted_review, color='red', label=f'Regression Line\ny = {slope:.2f}x + {intercept:.2f}')
plt.xlabel('User Review')
plt.ylabel('Meta Review')
plt.title('Scatterplot with Regression Line')
plt.legend()
plt.show()