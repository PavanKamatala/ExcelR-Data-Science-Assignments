"""
Created on Tue Nov 21 19:25:53 2023
"""
#Q7) Calculate Mean, Median, Mode, Variance, Standard Deviation, Range &     comment about the values / draw inferences, for the given data set
#-	For Points, Score, Weigh>
#Find Mean, Median, Mode, Variance, Standard Deviation, and Range and also Comment about the values/ Draw some inferences.

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
# Read the CSV file
df = pd.read_csv('Q7.csv')

# Extract columns
points = df['Points']
score = df['Score']
Weigh = df['Weigh']

# Mean
mean_points = np.mean(points)
mean_score = np.mean(score)
mean_Weigh = np.mean(Weigh)

# Median
median_points = np.median(points)
median_score = np.median(score)
median_Weigh = np.median(Weigh)

# Mode
mode_points = stats.mode(points).mode[0]
mode_score = stats.mode(score).mode[0]
mode_Weigh = stats.mode(Weigh).mode[0]

# Variance
variance_points = np.var(points)
variance_score = np.var(score)
variance_Weigh = np.var(Weigh)

# Standard Deviation
std_dev_points = np.std(points)
std_dev_score = np.std(score)
std_dev_Weigh = np.std(Weigh)

# Range
range_points = np.ptp(points)
range_score = np.ptp(score)
range_Weigh = np.ptp(Weigh)

# Output the results
print("Points: Mean =", mean_points, " Median =", median_points, " Mode =", mode_points,
      " Variance =", variance_points, " Standard Deviation =", std_dev_points, " Range =", range_points)

print("Score: Mean =", mean_score, " Median =", median_score, " Mode =", mode_score,
      " Variance =", variance_score, " Standard Deviation =", std_dev_score, " Range =", range_score)

print("Weigh: Mean =", mean_Weigh, " Median =", median_Weigh, " Mode =", mode_Weigh,
      " Variance =", variance_Weigh, " Standard Deviation =", std_dev_Weigh, " Range =", range_Weigh)

# Draw Histograms
plt.figure(figsize=(15, 5))

# Histogram for Points
plt.subplot(1, 3, 1)
plt.hist(points, color='blue')
plt.title('Histogram for Points')
plt.xlabel('Points')
plt.ylabel('Frequency')

# Histogram for Score
plt.subplot(1, 3, 2)
plt.hist(score, color='blue')
plt.title('Histogram for Score')
plt.xlabel('Score')
plt.ylabel('Frequency')

# Histogram for Weight
plt.subplot(1, 3, 3)
plt.hist(Weigh, color='blue')
plt.title('Histogram for Weight')
plt.xlabel('Weight')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()

'''
import pandas as pd
df=pd.read_csv("Q7.csv")
list(df)
df.head()
df.shape
df.dtypes
df

df[['Points','Score','Weigh']].min()
df[['Points','Score','Weigh']].max()
df[['Points','Score','Weigh']].mean()
df[['Points','Score','Weigh']].median()
df[['Points','Score','Weigh']].mode()
df[['Points','Score','Weigh']].std()
df[['Points','Score','Weigh']].var()
df[['Points','Score','Weigh']].sum()

df['Points'].max()-df['Points'].min()
df['Score'].max()-df['Score'].min()
df['Weigh'].max()-df['Weigh'].min()
import numpy as np
np.percentile(df[['Points','Score','Weigh']],25)
np.percentile(df[['Points','Score','Weigh']],50)
np.percentile(df[['Points','Score','Weigh']],75)

#To operate above all calculations in one time

df[['Points', 'Score', 'Weigh']].describe()

df[['Points','Score','Weigh']].corr()
df.corr()

#histogram

import matplotlib.pyplot as plt

df['Points'].hist()
df['Points'].skew()

df['Score'].hist()
df['Score'].skew()

df['Weigh'].hist()
df['Weigh'].skew()
'''
#==============================================================================
#==============================================================================
#9)a) Calculate Skewness, Kurtosis & draw inferences on the following data
#Cars speed and distance 
#Use Q9_a.csv
'''
import pandas as pd
from scipy.stats import skew, kurtosis

# Read the CSV file into a pandas DataFrame
data = pd.read_csv('Q9_a.csv')

# Extract the relevant columns (assuming the columns are named 'speed' and 'distance')
speed_data = data['speed']
dist_data = data['dist']

# Calculate skewness and kurtosis
speed_skew = skew(speed_data)
speed_kurt = kurtosis(speed_data)

dist_skew = skew(dist_data)
dist_kurt = kurtosis(dist_data)

# Print the results
print("Speed Skewness:", speed_skew)
print("Speed Kurtosis:", speed_kurt)
print("\nDistance Skewness:", dist_skew)
print("Distance Kurtosis:", dist_kurt)
'''


import pandas as pd
df=pd.read_csv("Q9_a.csv")
list(df)
df

import numpy as np
from scipy.stats import skew, kurtosis
import matplotlib.pyplot as plt
import seaborn as sns

df= np.random.randn(1000)  # Example dataset

skewness_value = skew(df)
kurtosis_value = kurtosis(df)
skewness_value 
kurtosis_value

plt.figure(figsize=(12, 6))

# Histogram
sns.histplot(df, kde=True, bins=30, color='blue', label='Data Distribution')

# Add vertical lines for mean, skewness, and kurtosis
plt.axvline(np.mean(df), color='red', linestyle='dashed', linewidth=2, label='Mean')
plt.axvline(skewness_value, color='green', linestyle='dashed', linewidth=2, label='Skewness')
plt.axvline(kurtosis_value, color='purple', linestyle='dashed', linewidth=2, label='Kurtosis')

plt.legend()
plt.show()

#==============================================================================
#9)b)SP and Weight(WT) Use Q9_b.csv

import pandas as pd
df=pd.read_csv("Q9_b.csv")
list(df)
df

import numpy as np
from scipy.stats import skew, kurtosis
import matplotlib.pyplot as plt
import seaborn as sns

df= np.random.randn(1000)  # Example dataset

skewness_value = skew(df)
kurtosis_value = kurtosis(df)
skewness_value 
kurtosis_value

plt.figure(figsize=(12, 6))

# Histogram
sns.histplot(df, kde=True, bins=30, color='blue', label='Data Distribution')

# Add vertical lines for mean, skewness, and kurtosis
plt.axvline(np.mean(df), color='red', linestyle='dashed', linewidth=2, label='Mean')
plt.axvline(skewness_value, color='green', linestyle='dashed', linewidth=2, label='Skewness')
plt.axvline(kurtosis_value, color='purple', linestyle='dashed', linewidth=2, label='Kurtosis')

plt.legend()
plt.show()
#==============================================================================

#==============================================================================
#11)
import numpy as np
from scipy.stats import norm

# Given data
sample_size = 2000
sample_mean = 200
sample_std_dev = 30

# Degrees of freedom for a normal distribution
degrees_of_freedom = sample_size - 1

# Confidence levels
confidence_levels = [0.94, 0.98, 0.96]

# Calculate the standard error of the mean (SEM)
sem = sample_std_dev / np.sqrt(sample_size)

# Calculate critical values for the confidence intervals
critical_values = [norm.ppf((1 + level) / 2) for level in confidence_levels]

# Calculate margin of error for each confidence level
margin_of_errors = [critical_value * sem for critical_value in critical_values]

# Calculate confidence intervals
confidence_intervals = [
    (sample_mean - margin_of_error, sample_mean + margin_of_error)
    for margin_of_error in margin_of_errors
]

# Print the results
for level, interval in zip(confidence_levels, confidence_intervals):
    print(f"{level * 100}% Confidence Interval: ({interval[0]}, {interval[1]})")
#==============================================================================

#==============================================================================

#12)Below are the scores obtained by a student in tests 
#34,36,36,38,38,39,39,40,40,41,41,41,41,42,42,45,49,56
#1)	Find mean, median, variance, standard deviation.
#2)	What can we say about the student marks? 


import numpy as np

# Given scores
scores = np.array([34,36,36,38,38,39,39,40,40,41,41,41,41,42,42,45,49,56])

# Calculate mean, median, variance, and standard deviation
mean_score = np.mean(scores)
median_score = np.median(scores)
variance_score = np.var(scores)
std_deviation_score = np.std(scores)

# Print the results
print("Mean:", mean_score)
print("Median:", median_score)
print("Variance:", variance_score)
print("Standard Deviation:", std_deviation_score)

import numpy as np
import pandas as pd
d1=[34,36,36,38,38,39,39,40,40,41,41,41,41,42,42,45,49,56]
d1
18*60
np.sum(d1)
np.mean(d1)
np.median(d1)
np.std(d1)
np.var(d1)

(738/1080)*100

#==============================================================================

#==============================================================================

#20)Calculate probability from the given dataset for the below cases
import pandas as pd

# Load the dataset
cars_data = pd.read_csv('Cars (1).csv')

# Extract the MPG column
MPG = cars_data['MPG']

# a. P(MPG > 38)
prob_a = len(MPG[MPG > 38]) / len(MPG)

# b. P(MPG < 40)
prob_b = len(MPG[MPG < 40]) / len(MPG)

# c. P(20 < MPG < 50)
prob_c = len(MPG[(MPG > 20) & (MPG < 50)]) / len(MPG)

# Print the probabilities
print("a. P(MPG > 38):", prob_a)
print("b. P(MPG < 40):", prob_b)
print("c. P(20 < MPG < 50):", prob_c)

#==============================================================================

#==============================================================================

#normal disdtribution
#Q21.a
   
#Q21.b
import pandas as pd
df = pd.read_csv("wc-at.csv")
df
import pandas as pd
df=pd.read_csv("Cars (1).csv")
df
df['MPG'].describe()

from scipy.stats import shapiro

stat, p_value = shapiro(df['MPG'])

alpha = 0.05
if p_value > alpha:
    print("The data follows a normal distribution ")
else:
    print("The data does not follow a normal distribution")
#21b)    
list(df)
from scipy import stats

zcal,pvalue = stats.ttest_ind(df['AT'], df['Waist'])

print("Z calculated value:", zcal.round(3))
print("pvalue value:", pvalue.round(3))

if pvalue < 0.05:
    print("Ho is rejected and H1 is accepted.")
    print("The data does not follow a normal distribution")
else:
    print("H1 is rejected and Ho is accepted.")    
    print("The data follows a normal distribution ")
#==============================================================================

#==============================================================================
#22) Calculate the Z scores of  90% confidence interval,94% confidence interval, 60% confidence interval 

import pandas as pd
from scipy.stats import norm

# Load the dataset (assuming 'Cars.csv' is your dataset)
df = pd.read_csv('Cars (1).csv')

# Extract the MPG column
mpg_data = df['MPG']

# Calculate mean and standard deviation
mean_mpg = mpg_data.mean()
std_dev_mpg = mpg_data.std()

# Confidence intervals
confidence_intervals = [0.90, 0.94, 0.60]

# Calculate Z-scores for each confidence interval
for confidence_interval in confidence_intervals:
    # Calculate the critical Z-value
    z_score = norm.ppf((1 + confidence_interval) / 2)
  
    print(f'{confidence_interval * 100}% Confidence Interval:')
    print(f'   Z-score: {z_score:.4f}')
#==============================================================================

#==============================================================================
#23)Calculate the t scores of 95% confidence interval, 96% confidence interval, 99% confidence interval for sample size of 25

import pandas as pd
from scipy.stats import t

# Load the dataset (assuming 'Cars.csv' is your dataset)
df = pd.read_csv('Cars (1).csv')

# Extract the MPG column
mpg_data = df['MPG']

# Sample size
sample_size = 25

# Degrees of freedom for a sample: df = sample_size - 1
degrees_of_freedom = sample_size - 1

# Confidence intervals
confidence_intervals = [0.95, 0.96, 0.99]

# Calculate t-scores for each confidence interval
for confidence_interval in confidence_intervals:
    # Calculate the critical t-value
    t_score = t.ppf((1 + confidence_interval) / 2, degrees_of_freedom)
    
    print(f'{confidence_interval * 100}% Confidence Interval:')
    print(f'   t-score: {t_score:.4f}')
#==============================================================================

#==============================================================================
'''
Q 24)   A Government  company claims that an average light bulb lasts 270 days.
    A researcher randomly selects 18 bulbs for testing. The sampled bulbs last an average of 260 days,
    with a standard deviation of 90 days. If the CEO's claim were true, what is the probability that 18 randomly
    selected bulbs would have an average life of no more than 260 days
'''

from scipy.stats import t

# Given data
population_mean = 270  # claimed by the company
sample_mean = 260
sample_std_dev = 90
sample_size = 18

# Calculate the t-score
t_score = (sample_mean - population_mean) / (sample_std_dev / (sample_size ** 0.5))

# Degrees of freedom
df = sample_size - 1

# Calculate the probability using the cumulative distribution function (CDF)
probability = t.cdf(t_score, df)

# Print the result
print("Probability that 18 randomly selected bulbs would have an average life of no more than 260 days:", probability)

#==============================================================================








