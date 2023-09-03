import pandas as pd
import matplotlib.pyplot as plt
import math
import random

data = pd.read_csv('Assignment\games.csv')
meta1 = data['meta_score']
user1 = data['user_review']

#using min max formula
meta = meta1-0/100-0
user = user1-0/10-0

def meta_stats(data):
    mean = meta.mean()
    median = meta.median()
    var = meta.var()
    std = meta.std()
    occur = meta.count()
    mode = meta.mode().iloc[0]

    print(f"Statistics for Meta score are as folows :")
    print(f"Mean : {mean}")
    print(f"Median : {median}")
    print(f"Mode : {mode}")
    print(f"Occurences : {occur}")
    print(f"Variance : {var}")
    print(f"Standard Deviation : {std}")
meta_stats(data=meta)

def user_stats(data1):
    mean = user.mean()
    median = user.median()
    var = user.var()
    std = user.std()
    occur = user.count()
    mode = user.mode().iloc[0]

    print(f"\n \nStatistics for User review score are as folows :")
    print(f"Mean : {mean}")
    print(f"Median : {median}")
    print(f"Mode : {mode}")
    print(f"Occurences : {occur}")
    print(f"Variance : {var}")
    print(f"Standard Deviation : {std}")
user_stats(data1=user)

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

def normal_user():
    mean = user1.mean()
    std = user1.std()
    x = user1.sample().values[0]
    pi = 3.14
    e = 2.71
    power = -1/2*((x - mean)/std)**2
    num = math.exp(e**power)
    sq = math.sqrt(2*pi)
    deno = std*sq
    print(f"Normal distribution value for user score {x} is: {num/deno}")
normal_user()

#to find bins
# bins = 1 + math.log2(17440)
# print(bins)

plt.hist(meta1, bins = 15, edgecolor = 'black')
plt.xlabel('Score out of 1')
plt.ylabel('Number of games')
plt.title('Meta Score')
plt.show()

plt.hist(user1, bins = 15, edgecolor = 'black')
plt.xlabel('Score out of 1')
plt.ylabel('Number of games')
plt.title('User review score')
plt.show()