import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, log_loss, confusion_matrix

pd.set_option('future.no_silent_downcasting', True)
dataframe = pd.read_csv('credit_risk_dataset.csv')
dataframe = dataframe.dropna()
dataframe['cb_person_default_on_file'] = dataframe['cb_person_default_on_file'].replace({'Y': 1, 'N': 0})

X = dataframe[['person_age', 'person_income', 'person_emp_length', 'loan_amnt', 'loan_int_rate', 'loan_percent_income', 'cb_person_default_on_file', 'cb_person_cred_hist_length']]
y = dataframe['loan_status']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=86)

model = GaussianNB()
model.fit(X_train, y_train)

# Predictions on training data
y_train_pred = model.predict(X_train)
y_train_prob = model.predict_proba(X_train)

# Predictions on testing data
y_test_pred = model.predict(X_test)
y_test_prob = model.predict_proba(X_test)

# Accuracy
accuracy_train = accuracy_score(y_train, y_train_pred)
accuracy_test = accuracy_score(y_test, y_test_pred)

# Sensitivity (Recall)
sensitivity_train = recall_score(y_train, y_train_pred)
sensitivity_test = recall_score(y_test, y_test_pred)

# Specificity
tn_train, fp_train, fn_train, tp_train = confusion_matrix(y_train, y_train_pred).ravel()
specificity_train = tn_train / (tn_train + fp_train)

tn_test, fp_test, fn_test, tp_test = confusion_matrix(y_test, y_test_pred).ravel()
specificity_test = tn_test / (tn_test + fp_test)

# F1 Score
f1_train = f1_score(y_train, y_train_pred)
f1_test = f1_score(y_test, y_test_pred)

# Log Loss
log_loss_train = log_loss(y_train, y_train_prob)
log_loss_test = log_loss(y_test, y_test_prob)

print(f'Number of data instances: {dataframe.shape[0]}')
print(f'Number of input features: {dataframe.shape[1] - 1}')

print(f"METRIC\tTRAIN\tTEST")
print(f'Acc\t{accuracy_train:.2f}\t{accuracy_test:.2f}')
print(f'Sens\t{sensitivity_train:.2f}\t{sensitivity_test:.2f}')
print(f'Spec\t{specificity_train:.2f}\t{specificity_test:.2f}')
print(f'F1\t{f1_train:.2f}\t{f1_test:.2f}')
print(f'LL\t{log_loss_train:.2f}\t{log_loss_test:.2f}')

#y_pred = model.predict(X_test)
#accuracy = accuracy_score(y_test, y_test_pred)
#print(f'Accuracy: {accuracy:.2f}')