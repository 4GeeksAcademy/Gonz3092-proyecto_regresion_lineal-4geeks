import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

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


scaler =  MinMaxScaler()

X_train_scal = pd.DataFrame(scaler.fit_transform(X_train), columns=num_variables)
X_test_scal = pd.DataFrame(scaler.transform(X_test), columns=num_variables)