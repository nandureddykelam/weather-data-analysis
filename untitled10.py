import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

file_path = r"C:\Users\nandu\Downloads\weather_dataset_1000_rows_10_columns.csv"
df = pd.read_csv(file_path)

df.columns = df.columns.str.strip()
df = df.apply(pd.to_numeric, errors='ignore')
df.dropna(inplace=True)
df.drop_duplicates(inplace=True)

print("Dataset Loaded")
print(df.head())

def temp_category(temp):
    if temp < 20:
        return "Low"
    elif temp < 35:
        return "Medium"
    else:
        return "High"

df['Temp_Category'] = df['Temperature_C'].apply(temp_category)

df['Wind_Category'] = pd.cut(
    df['Wind_Speed_km_h'],
    bins=[0, 10, 25, 50],
    labels=['Low', 'Medium', 'High']
)

print(df['Temp_Category'].value_counts())

sns.set_theme(style="whitegrid")

plt.figure(figsize=(8, 5))
ax = sns.countplot(x='Temp_Category', data=df, palette='viridis', order=['Low', 'Medium', 'High'])
for container in ax.containers:
    ax.bar_label(container, fmt='%d', padding=3)
plt.title("Count of Temperature Categories")
plt.ylabel("Count")
plt.show()

plt.figure(figsize=(8, 5))
sns.kdeplot(data=df, x='Temperature_C', hue='Wind_Category', fill=True, alpha=0.5, palette='Set1')
plt.title("Temperature Distribution Grouped by Wind Category")
plt.xlabel("Temperature (°C)")
plt.show()

plt.figure(figsize=(10, 8))
corr = df.select_dtypes(include=[np.number]).corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, cmap='coolwarm', fmt=".2f",
            linewidths=.5, cbar_kws={"shrink": .8})
plt.title("Upper-Triangle Correlation Heatmap")
plt.show()

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
temp_counts = df['Temp_Category'].value_counts()
axes[0].pie(temp_counts, labels=temp_counts.index, autopct='%1.1f%%',
            colors=sns.color_palette("pastel"), startangle=90,
            wedgeprops={'edgecolor': 'white'})
axes[0].set_title('Temperature Category Distribution')

wind_counts = df['Wind_Category'].value_counts()
axes[1].pie(wind_counts, labels=wind_counts.index, autopct='%1.1f%%',
            colors=sns.color_palette("Set3"), startangle=90,
            wedgeprops={'edgecolor': 'white'})
axes[1].set_title('Wind Category Distribution')
plt.tight_layout()
plt.show()

plt.figure(figsize=(9, 6))
ax2 = sns.countplot(data=df, x='Temp_Category', hue='Wind_Category',
                    palette='muted', order=['Low', 'Medium', 'High'])
for container in ax2.containers:
    ax2.bar_label(container, padding=3)
plt.title("Temperature Categories grouped by Wind Category")
plt.ylabel("Count")
plt.legend(title='Wind Category')
plt.show()

plt.figure(figsize=(10, 6))
sns.violinplot(x='Temp_Category', y='Humidity_%', data=df,
               palette='pastel', inner=None, order=['Low', 'Medium', 'High'])
sns.swarmplot(x='Temp_Category', y='Humidity_%', data=df,
              color='k', alpha=0.5, size=3,
              order=['Low', 'Medium', 'High'])
plt.title("Humidity Distribution across Temperature Categories")
plt.show()

sns.jointplot(x='Temperature_C', y='Wind_Speed_km_h',
              data=df, kind='reg', color='m',
              height=7, scatter_kws={'alpha': 0.3})
plt.suptitle("Temperature vs Wind Speed with Regression Line", y=1.02)
plt.show()

plt.figure(figsize=(8, 5))
sns.boxenplot(x='Wind_Category', y='Temperature_C',
              data=df, palette='Set2')
plt.title("Boxen Plot of Temperature by Wind Category")
plt.show()

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
cols_to_plot = numeric_cols[:4] if len(numeric_cols) > 4 else numeric_cols
df_subset = df[cols_to_plot + ['Temp_Category']]

sns.pairplot(df_subset, hue='Temp_Category',
             palette='husl', corner=True)
plt.suptitle("Pairplot of Numeric Variables", y=1.02)
plt.show()

low_wind_temp = df[df['Wind_Category'] == 'Low']['Temperature_C'].dropna()
high_wind_temp = df[df['Wind_Category'] == 'High']['Temperature_C'].dropna()

t_stat, p_val_t = stats.ttest_ind(low_wind_temp, high_wind_temp, equal_var=False)
print("T-Test:", t_stat, p_val_t)

stat_shapiro, p_val_shapiro = stats.shapiro(df['Temperature_C'].dropna())
print("Shapiro Test:", stat_shapiro, p_val_shapiro)

contingency_table = pd.crosstab(df['Temp_Category'], df['Wind_Category'])
chi2, p_val_chi, dof, expected = stats.chi2_contingency(contingency_table)
print("Chi-Square:", chi2, p_val_chi)

numeric_df = df.select_dtypes(include=[np.number]).dropna()
numeric_df['Intercept'] = 1

vif_data = pd.DataFrame()
vif_data["Feature"] = numeric_df.columns
vif_data["VIF"] = [variance_inflation_factor(numeric_df.values, i)
                   for i in range(len(numeric_df.columns))]
vif_data = vif_data[vif_data['Feature'] != 'Intercept']
print(vif_data)

target = 'Temperature_C'

X = df.select_dtypes(include=[np.number]).drop(columns=[target], errors='ignore')
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("R2 Score:", r2_score(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))

plt.figure(figsize=(8, 5))
plt.scatter(y_test, y_pred, alpha=0.6, color='dodgerblue', edgecolor='k')
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()], 'r--', lw=2)
plt.title("Actual vs Predicted")
plt.xlabel("Actual Temperature")
plt.ylabel("Predicted Temperature")
plt.show()

plt.figure(figsize=(8, 5))
residuals = y_test - y_pred
sns.residplot(x=y_pred, y=residuals, lowess=True,
              scatter_kws={'alpha': 0.5, 'color': 'teal', 'edgecolor': 'k'},
              line_kws={'color': 'red', 'lw': 2})
plt.title("Residual Plot")
plt.xlabel("Predicted")
plt.ylabel("Residuals")
plt.axhline(0, color='black', linestyle='--')
plt.show()

print("FINAL PROJECT COMPLETED SUCCESSFULLY")