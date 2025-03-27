import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score



url = 'https://raw.githubusercontent.com/4GeeksAcademy/linear-regression-project-tutorial/main/medical_insurance_cost.csv'

data = pd.read_csv(url)
data = data.drop_duplicates().reset_index(drop=True)


data['sex_n'] = pd.factorize(data['sex'])[0]
data['smoker_n'] = pd.factorize(data['smoker'])[0]
data['region_n'] = pd.factorize(data['region'])[0]

num_variables = ["age", "bmi", "children", "region_n", "sex_n", "smoker_n"]

X = data.drop("charges", axis = 1)[num_variables]
y = data["charges"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

scaler = MinMaxScaler()

X_train_scal = pd.DataFrame(scaler.fit_transform(X_train[num_variables]), columns=num_variables)
X_test_scal = pd.DataFrame(scaler.transform(X_test[num_variables]), columns=num_variables)

selection_model = SelectKBest(score_func=f_regression, k=4)

selection_model.fit(X_train_scal, y_train)

ix = selection_model.get_support()
X_train_sel = pd.DataFrame(selection_model.transform(X_train_scal), columns=X_train.columns[ix])
X_test_sel = pd.DataFrame(selection_model.transform(X_test_scal), columns=X_train.columns[ix])

model = LinearRegression()
model.fit(X_train_sel, y_train)

print(f"Intercepto (a): {model.intercept_}")
print(f"Coeficientes (b): {model.coef_}")

y_pred = model.predict(X_test_sel)

print(f"MSE: {mean_squared_error(y_test, y_pred)}")
print(f"R2 Score: {r2_score(y_test, y_pred)}")