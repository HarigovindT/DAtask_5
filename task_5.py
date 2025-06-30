import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv(r"C:\Users\harig\Downloads\titanic\test.csv")

# Basic info
print("Data Info:")
print(df.info())

# Summary statistics
print("\nSummary Statistics:")
print(df.describe(include='all'))

# Missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Value counts for categorical columns
for col in df.select_dtypes(include='object').columns:
    print(f"\nValue counts for {col}:\n{df[col].value_counts()}")

# Histogram - Age
plt.figure(figsize=(6,4))
df['Age'].hist(bins=20, edgecolor='black')
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Count')
plt.show()

# Boxplot - Age vs Pclass
plt.figure(figsize=(6,4))
sns.boxplot(x='Pclass', y='Age', data=df)
plt.title('Age by Passenger Class')
plt.show()

# Scatterplot - Age vs Fare
plt.figure(figsize=(6,4))
sns.scatterplot(data=df, x='Age', y='Fare', hue='Sex')
plt.title('Fare vs Age')
plt.show()

# Heatmap - Correlation
plt.figure(figsize=(6,4))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()
