# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
url = 'https://web.stanford.edu/class/archive/cs/cs109/cs109.1166/stuff/titanic.csv'
data = pd.read_csv(url)

# Display the first few rows
print(data.head())

# Check the structure of the dataset
print(data.info())

# Summary statistics
print(data.describe())

# Handling missing values
data = data.dropna(subset=['Age'])  # Drop rows where 'Age' is missing

# Remove duplicates
data = data.drop_duplicates()

# Correct data types if necessary
# data['column_name'] = data['column_name'].astype('desired_dtype')

# Univariate Analysis

# Distribution of a numerical variable
plt.figure(figsize=(10, 6))
sns.histplot(data['Age'], kde=True)
plt.title('Distribution of Age')
plt.show()

# Box plot to identify outliers
plt.figure(figsize=(10, 6))
sns.boxplot(x=data['Age'])
plt.title('Box plot of Age')
plt.show()

# Bivariate Analysis

# Scatter plot between two numerical variables
plt.figure(figsize=(10, 6))
sns.scatterplot(x=data['Age'], y=data['Fare'])
plt.title('Scatter plot between Age and Fare')
plt.show()

# Correlation matrix
plt.figure(figsize=(12, 8))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# Multivariate Analysis

# Pair plot for multivariate analysis
plt.figure(figsize=(12, 12))
sns.pairplot(data[['Age', 'Fare', 'Pclass', 'Survived']])
plt.title('Pair plot of the Dataset')
plt.show()

# Summary and Insights
print("Summary and Insights:")
print("1. Distribution of Age: The age distribution is slightly right-skewed.")
print("2. Outliers: There are some outliers in the age and fare variables.")
print("3. Correlations: Age and Fare do not show a strong correlation. Fare is correlated with class.")
