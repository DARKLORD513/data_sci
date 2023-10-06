import pandas as pd
import matplotlib.pyplot as plt
import math
import numpy as np
from sklearn.linear_model import LinearRegression

data = pd.read_csv('stackoverflow_full.csv')
#meta1 = data['PreviousSalary']
meta1 = data['ComputerSkills']
meta2 = data['YearsCodePro']


def meta_stats(data):
    mean = data.mean()
    median = data.median()
    var = data.var()
    std = data.std()
    occur = data.count()
    mode = data.mode().iloc[0]

    print(f"Statistics for Employee Salary are as folows :")
    print(f"Mean : {mean}")
    print(f"Median : {median}")
    print(f"Mode : {mode}")
    print(f"Occurences : {occur}")
    print(f"Variance : {var}")
    print(f"Standard Deviation : {std}")
meta_stats(data=meta1)

#Normalization
#meta = (meta1-100024)/(224000-100024)
meta = (meta1-0)/(107-0)
years = (meta2-0)/(50-0)

def pearson():
    summation = sum(ai * bi for ai, bi in zip(meta, years))
    mean_a = meta.mean()
    mean_b = years.mean()
    n = 16647

    part1 = n*mean_a*mean_b
    numerator = summation - part1
    std_a = meta.std()
    std_b = years.std()
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
    final = "{:.2e}".format(num/deno)
    print(f"Normal distribution value for meta score {x} is: {final}")
normal_meta()

# #! 1d Histogram
# plt.hist(meta1, bins = 15, edgecolor = 'black')
# plt.xlabel('Salary')
# plt.ylabel('Number of employees')
# plt.title('Salary vs employees')
# plt.show()

# #! 2d Histogram
# plt.hist2d(meta2,meta1,bins=[16,6],cmap='Blues')
# plt.xlabel('Years of Coding')
# plt.ylabel('Salary')
# plt.title('Salary vs. Years of Coding')
# plt.colorbar()
# plt.show()

# #! Scatterplot
# plt.scatter(meta2, meta1, label='Data Points', color='blue', marker='o',s=1)
# plt.xlabel('X-axis Label')
# plt.ylabel('Y-axis Label')
# plt.title('Scatterplot Example')
# plt.legend()
# plt.show()

#! Regression line
plt.scatter(meta2, meta1, label='Data Points', color='blue', marker='o',s=1)
model = LinearRegression()
X = meta2.values.reshape(-1, 1)
y = meta1.values.reshape(-1, 1)
model.fit(X, y)

slope = model.coef_[0][0]
intercept = model.intercept_[0]
predicted_salary = model.predict(X)

plt.plot(meta2, predicted_salary, color='red', label=f'Regression Line\ny = {slope:.2f}x + {intercept:.2f}')
plt.xlabel('Years of Coding')
plt.ylabel('Salary')
plt.title('Scatterplot with Regression Line')
plt.legend()
plt.show()