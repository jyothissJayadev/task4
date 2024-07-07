import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

url = 'https://web.stanford.edu/class/archive/cs/cs109/cs109.1166/stuff/titanic.csv'
data = pd.read_csv(url)

print(data.head())
print(data.info())
print(data.describe())

data = data.dropna(subset=['Age'])
data = data.drop_duplicates()

plt.figure(figsize=(10, 6))
sns.histplot(data['Age'], kde=True)
plt.title('Distribution of Age')
plt.show()

plt.figure(figsize=(10, 6))
sns.boxplot(x=data['Age'])
plt.title('Box plot of Age')
plt.show()

plt.figure(figsize=(10, 6))
sns.scatterplot(x=data['Age'], y=data['Fare'])
plt.title('Scatter plot between Age and Fare')
plt.show()

numeric_data = data.select_dtypes(include='number')
plt.figure(figsize=(12, 8))
sns.heatmap(numeric_data.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

plt.figure(figsize=(12, 12))
sns.pairplot(numeric_data[['Age', 'Fare', 'Pclass', 'Survived']])
plt.title('Pair plot of the Dataset')
plt.show()

print("Summary and Insights:")
print("1. Distribution of Age: The age distribution is slightly right-skewed.")
print("2. Outliers: There are some outliers in the age and fare variables.")
print("3. Correlations: Age and Fare do not show a strong correlation. Fare is correlated with class.")
