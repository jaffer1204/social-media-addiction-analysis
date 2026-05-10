import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("dataset/Students Social Media Addiction.csv")

# Style
sns.set_style("darkgrid")

# 1. Most Used Platform
plt.figure(figsize=(8,5))
sns.countplot(x="Most_Used_Platform", data=df)

plt.title("Most Used Social Media Platforms")
plt.xticks(rotation=45)

plt.show()

# 2. Daily Usage Hours
plt.figure(figsize=(8,5))
sns.histplot(df["Avg_Daily_Usage_Hours"], bins=10)

plt.title("Daily Usage Hours Distribution")

plt.show()

# 3. Sleep vs Usage
plt.figure(figsize=(8,5))
sns.scatterplot(
    x="Avg_Daily_Usage_Hours",
    y="Sleep_Hours_Per_Night",
    data=df
)

plt.title("Sleep Hours vs Social Media Usage")

plt.show()

# 4. Correlation Heatmap
plt.figure(figsize=(10,6))

sns.heatmap(
    df.select_dtypes(include='number').corr(),
    annot=True,
    cmap="coolwarm"
)

plt.title("Correlation Heatmap")

plt.show()