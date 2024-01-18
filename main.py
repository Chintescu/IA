import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

df = pd.read_csv('D:\IA\Decision\HR-Em.csv')

print(df.head())

missing_values = df.isnull().sum()
print("\nMissing Values:")
print(missing_values)

duplicates = df.duplicated().sum()
print(f"\nNumber of Duplicates: {duplicates}")

summary_stats = df.describe()
print("\nSummary Statistics:")
print(summary_stats)

numeric_variables = ['DailyRate', 'HourlyRate', 'MonthlyIncome', 'JobSatisfaction']
df[numeric_variables].hist(bins=20, figsize=(12, 8))
plt.suptitle('Distribuția Variabilelor Numerice')
plt.show()

categorical_variables = ['BusinessTravel', 'EducationField', 'Gender', 'JobRole', 'MaritalStatus']
for var in categorical_variables:
    plt.figure(figsize=(10, 6))
    sns.countplot(x=var, data=df)
    plt.title(f'Distribuția variabilei {var}')
    plt.show()

correlation_matrix = df[numeric_variables].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Matricea de corelații')
plt.show()

plt.figure(figsize=(12, 6))
sns.barplot(x='MonthlyIncome', y='Education', data=df, estimator=sum)
plt.title('Salariu in functie de educatie')
plt.xlabel('MonthlyIncome')
plt.ylabel('Education')
plt.show()

plt.figure(figsize=(12, 6))
sns.barplot(x='MonthlyIncome', y='HourlyRate', data=df, estimator=sum)
plt.title('Salariu in functie de ore muncite')
plt.xlabel('MonthlyIncome')
plt.ylabel('HourlyRate')
plt.show()


df.columns = df.columns.str.lower()
X_reg = df[['MonthlyIncome', 'HourlyRate']]
y_reg = df['Age']
X_clf = df[['MonthlyIncome', 'HourlyRate']]
y_clf = df['JobSatisfaction']

X_reg_train, X_reg_test, y_reg_train, y_reg_test = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)
X_clf_train, X_clf_test, y_clf_train, y_clf_test = train_test_split(X_clf, y_clf, test_size=0.2, random_state=42)

#Regresie Liniara
regression_model = LinearRegression()
regression_model.fit(X_reg_train, y_reg_train)
regression_predictions = regression_model.predict(X_reg_test)
regression_mse = mean_squared_error(y_reg_test, regression_predictions)
print(f"Mean Squared Error (Linear Regression): {regression_mse}")

#Arbori de Decizie pentru Regresie
tree_reg_model = DecisionTreeRegressor(random_state=42)
tree_reg_model.fit(X_reg_train, y_reg_train)
tree_reg_predictions = tree_reg_model.predict(X_reg_test)
tree_reg_mse = mean_squared_error(y_reg_test, tree_reg_predictions)
print(f"Mean Squared Error (Decision Tree Regression): {tree_reg_mse}")

#Arbori de Decizie pentru Clasificare
tree_clf_model = DecisionTreeClassifier(random_state=42)
tree_clf_model.fit(X_clf_train, y_clf_train)
tree_clf_predictions = tree_clf_model.predict(X_clf_test)
accuracy = accuracy_score(y_clf_test, tree_clf_predictions)
print(f"Accuracy (Decision Tree Classification): {accuracy}")

#Naive Bayes pentru Clasificare
nb_model = GaussianNB()
nb_model.fit(X_clf_train, y_clf_train)
nb_predictions = nb_model.predict(X_clf_test)
accuracy = accuracy_score(y_clf_test, nb_predictions)
print(f"Accuracy (Naive Bayes): {accuracy}")

#K-Means pentru Clustering
kmeans_model = KMeans(n_clusters=3, random_state=42)
df['cluster'] = kmeans_model.fit_predict(X_reg)
plt.scatter(X_reg['MonthlyIncome'], X_reg['Age'], c=df['cluster'], cmap='viridis', s=50)
plt.title('K-Means Clustering')
plt.xlabel('MonthlyIncome')
plt.ylabel('Education')
plt.show()

#PCA (Principal Component Analysis)
pca = PCA(n_components=2)
pca_result = pca.fit_transform(X_reg)
df['pca_1'] = pca_result[:, 0]
df['pca_2'] = pca_result[:, 1]
plt.figure(figsize=(10, 6))
plt.scatter(df['pca_1'], df['pca_2'], c='blue', alpha=0.5)
plt.title('Principal Component Analysis (PCA)')
plt.xlabel('MonthlyIncome')
plt.ylabel('Education')
plt.show()

#Random Forest pentru Regresie
rf_reg_model = RandomForestRegressor(random_state=42)
rf_reg_model.fit(X_reg_train, y_reg_train)
rf_reg_predictions = rf_reg_model.predict(X_reg_test)
rf_reg_mse = mean_squared_error(y_reg_test, rf_reg_predictions)
print(f"Mean Squared Error (Random Forest Regression): {rf_reg_mse}")

#Random Forest pentru Clasificare
rf_clf_model = RandomForestClassifier(random_state=42)
rf_clf_model.fit(X_clf_train, y_clf_train)
rf_clf_predictions = rf_clf_model.predict(X_clf_test)
accuracy = accuracy_score(y_clf_test, rf_clf_predictions)
print(f"Accuracy (Random Forest Classification): {accuracy}")


