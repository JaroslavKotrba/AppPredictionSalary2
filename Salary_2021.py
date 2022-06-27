# SALARY APP 2021

import numpy as np
import pandas as pd
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 100)
import matplotlib.pyplot as plt
import seaborn as sn

import os
path = "/Users/HP/OneDrive/Documents/Python Anaconda/Flask_Salary_App/Salary_2021"
os.chdir(path)
os.listdir()

import warnings
warnings.filterwarnings("ignore")

# Import
df = pd.read_csv('survey_results_public_2021.csv')
df.columns
df.shape
df

# Select
df = df[['ConvertedCompYearly', 'Employment', 'Country', 'EdLevel', 'Age1stCode', 'YearsCode', 'YearsCodePro', 'OrgSize', 'OpSys', 'Age', 'LanguageHaveWorkedWith']]

# $Salary and NAs
df.rename({'ConvertedCompYearly':'Salary'}, axis=1, inplace=True)
df = df[df["Salary"].notnull()]
df = df.dropna()
df.isnull().sum()

# $Employment
df = df[df["Employment"] == "Employed full-time"]
df = df.drop("Employment", axis=1)
df.info()

# $Country
df['Country'].value_counts()

df['Country'] = df['Country'].str.replace('United Kingdom of Great Britain and Northern Ireland','UK')
df['Country'] = df['Country'].str.replace('United States of America','USA')
df['Country'] = df['Country'].str.replace('Iran, Islamic Republic of...', 'Iran')

def shorten_categories(categories, cutoff):
    categorical_map = {}
    for i in range(len(categories)):
        if categories.values[i] >= cutoff:
            categorical_map[categories.index[i]] = categories.index[i]
        else:
            categorical_map[categories.index[i]] = 'Other'
    return categorical_map

country_map = shorten_categories(df.Country.value_counts(), 50)
df['Country'] = df['Country'].map(country_map)
df.Country.value_counts()

df = df[df['Country'] != 'Other']

def viz(column):
    sum_row = pd.DataFrame(df[column].value_counts()).reset_index().rename(columns={'index':column, column:'Count'})

    plt.figure(figsize=(12,8))
    plt.bar(sum_row[column], sum_row['Count'])
    plt.xticks(rotation='vertical')

viz('Country');

# $Education
df['EdLevel'].value_counts()

def clean_education(x):
    if 'Bachelor’s degree' in x:
        return 'Bachelor’s degree'
    if 'Master’s degree' in x:
        return 'Master’s degree'
    if 'Professional degree' in x or 'Other doctoral' in x:
        return 'Post grad'
    return 'Less than a Bachelor'
df['EdLevel'] = df['EdLevel'].apply(clean_education)

viz('EdLevel');

# $Age1stCode
df['Age1stCode'].value_counts()

def clean_experience(x):
    if x == 'Older than 64 years':
        return 65
    if x == '55 - 64 years':
        return 55
    if x == '45 - 54 years':
        return 45
    if x == '35 - 44 years':
        return 35
    if x == '25 - 34 years':
        return 26
    if x == '18 - 24 years':
        return 20
    if x == '11 - 17 years':
        return 15
    if x == '5 - 10 years':
        return 6
    if x == 'Younger than 5 years':
        return 3
    return float(x)
df['Age1stCode'] = df['Age1stCode'].apply(clean_experience)

# $YearsCode
df['YearsCode'].value_counts()

def clean_experience(x):
    if x == 'More than 50 years':
        return 50
    if x == 'Less than 1 year':
        return 0.5
    return float(x)
df['YearsCode'] = df['YearsCode'].apply(clean_experience)

# $YearsCodePro
df['YearsCodePro'].value_counts()

def clean_experience(x):
    if x == 'More than 50 years':
        return 50
    if x == 'Less than 1 year':
        return 0.5
    return float(x)
df['YearsCodePro'] = df['YearsCodePro'].apply(clean_experience)

# $OrgSize
df['OrgSize'].value_counts()

sum_row = df.groupby('OrgSize').agg({'Salary':'mean'}).reset_index().round().sort_values('Salary', ascending=False); sum_row

plt.figure(figsize=(12,8))
plt.bar(sum_row['OrgSize'], sum_row['Salary'])
plt.xticks(rotation='vertical')

# $OpSys
df['OpSys'].value_counts()

def clean_os(x):
    out = []
    for i in x:
        if pd.isnull(i): # in case of numeric if np.isnan(i):
            out.append(float("nan"))
        elif i == 'Windows':
            out.append("Windows")
        elif i == 'Linux-based':
            out.append("Linux")
        elif i == 'MacOS':
            out.append("MacOS")
        elif i == 'Windows Subsystem for Linux (WSL)':
            out.append("Windows")
        else:
            out.append("Other")
    return out

df["OpSys"] = clean_os(df["OpSys"])

viz('OpSys');

# $Age
df['Age'].value_counts()

def clean_age(x):
    if x == '65 years or older':
        return '65+'
    if x == '55-64 years old':
        return '55-64'
    if x == '45-54 years old':
        return '45-54'
    if x == '35-44 years old':
        return '35-44'
    if x == '25-34 years old':
        return '25-34'
    if x == '18-24 years old':
        return '18-24'
    if x == 'Under 18 years old':
        return 'Under 18'
    if x == 'Prefer not to say':
        return 'Do not want to say!'
    return float(x)
df['Age'] = df['Age'].apply(clean_age)

# $LanguageHaveWorkedWith
df['LanguageHaveWorkedWith'].value_counts()

df = pd.concat([df, df['LanguageHaveWorkedWith'].str.get_dummies(sep=';')], axis=1); df
df.drop(columns=['LanguageHaveWorkedWith'], inplace=True)

sum_row = pd.DataFrame({'Count':df.iloc[:, 9:].sum()}).sort_values('Count', ascending=False).reset_index()
sum_row.index = np.arange(1, len(sum_row) + 1)
sum_row.rename(columns={'index':'Language'}, inplace=True); sum_row

plt.figure(figsize=(12,8))
plt.bar(sum_row['Language'], sum_row['Count'])
plt.xticks(rotation='vertical');

# Changes
df = df[(df['Salary'] <= 500000) & (df['Salary'] >= 10000)].sort_values('Salary', ascending=False)

# Droping columns (multicollinearity)
df.drop(columns=['Age1stCode', 'YearsCode'], inplace=True)

# Saving df for app.py
df.to_csv("survey_clean.csv", index=False)

# Vizualisation by country
fig, ax = plt.subplots(1,1,figsize=(16,12))
df.boxplot('Salary', 'Country', ax=ax, color=dict(boxes='b', whiskers='b', medians='r', caps='b'))
plt.suptitle('Salary (USD) by Country')
plt.ylabel('Salary')
plt.xticks(rotation=90)
plt.show()

# Vizualisation by country plotly
import plotly.express as px
fig = px.box(df.sort_values('Country'), x="Country", y="Salary", 
                                template='simple_white',
                                title=f"<b>Global</b> - Salary based on country (USD)")
fig.update_xaxes(tickangle=90, tickmode = 'array', tickvals = df.sort_values('Country')['Country'])
fig.show()

# Vizualisation by experience plotly
import plotly.express as px

object = "Czech Republic"

country = df[df['Country']==object]
country['EdLevel'] = pd.Categorical(country['EdLevel'], ['Less than a Bachelor', 'Bachelor’s degree', 'Master’s degree', 'Post grad'])
country = country.sort_values('EdLevel')

fig = px.scatter(country, x="YearsCodePro", y="Salary", trendline="ols", color="EdLevel", symbol='EdLevel', opacity=0.8,
                                marginal_x="histogram", 
                                marginal_y="rug",
                                template='simple_white', hover_data=['Salary', 'YearsCodePro', 'Age', 'OrgSize'],
                                title=f" <b>{object}</b> - Salary based on experience & education (USD)")
fig.update_traces(marker=dict(size=8, line=dict(width=1, color='black')), selector=dict(mode='markers'))
fig.show()

# Table with mean and median
table = df.groupby('Country').mean().round()
table = table.rename(columns={"YearsCodePro" : "YearsCodeProMean", "Salary" : "SalaryMean"})
table = table.join(df.groupby('Country').median()[["YearsCodePro", 'Salary']])
table = table.rename(columns={"YearsCodePro" : "YearsCodeProMedian", "Salary" : "SalaryMedian"})
table[['SalaryMean','SalaryMedian']].sort_values('SalaryMean', ascending=False)

# Outliers categorical
def remove_outliers(df):
    out = pd.DataFrame()
    for key, subset in df.groupby('Country'):
        m = np.mean(subset.Salary)
        st = np.std(subset.Salary)
        reduced_df = subset[(subset.Salary>(m-st)) & (subset.Salary<=(m+st))]
        out = pd.concat([out, reduced_df], ignore_index=True)
    return out
df = remove_outliers(df)

# ---------------------------------------------------------------------------------------------------------------
# Modeling pipe based -------------------------------------------------------------------------------------------

# Splitting
X = df.drop(columns=['Salary']);
y = df.Salary; y

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Preprocessing
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder
column_trans = make_column_transformer((OneHotEncoder(sparse=False), ['Country', 'EdLevel', 'OrgSize', 'OpSys', 'Age']), # non-numeric
                                        remainder='passthrough')

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

# ---------------------------------------------------------------------------------------------------------------
# MODEL Linear regression ---------------------------------------------------------------------------------------

import timeit
start = timeit.default_timer()

from sklearn.linear_model import LinearRegression
model = LinearRegression(normalize=True)

from sklearn.pipeline import make_pipeline
pipe = make_pipeline(column_trans, scaler, model)
pipe.fit(X_train, y_train)

y_pred = pipe.predict(X_test)

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
outcome = pd.DataFrame({'y_test':y_test, 'y_pred':y_pred})
outcome['difference'] = outcome['y_test'] - outcome['y_pred']
outcome['difference_percentage'] = round(outcome.difference/(outcome.y_test/100),6)

print('Linear:')
print('PROC: ', round(outcome.difference_percentage.abs().mean(),2),'%')
print('MAE: ', round(mean_absolute_error(y_test, y_pred),4))
print('RMSE: ', round(np.sqrt(mean_squared_error(y_test, y_pred)),4))
print('R2:', round(r2_score(y_test, y_pred),4))

stop = timeit.default_timer()
print('Time: ', stop - start)

# Sample
X_sample = np.array(["Czech Republic", "Master’s degree", "2", "10,000 or more employees", "MacOS", "25-34",
                      "0",       "0",         "1",     "0",  "0",  "0",    "0",      "0",       "0",      "0",     "0",      "0",      "0",    "0",  "0",    "0",       "1",       "0",      "0",       "0",       "0",     "0",     "0",      "0",      "0",         "0",       "0",    "0",       "0",        "1",   "1",  "0",    "0",    "1",    "0",    "0",        "0",       "0"])
                    # 'APL', 'Assembly', 'Bash/Shell', 'C', 'C#', 'C++', 'COBOL', 'Clojure', 'Crystal', 'Dart', 'Delphi', 'Elixir', 'Erlang', 'F#', 'Go', 'Groovy', 'HTML/CSS', 'Haskell', 'Java', 'JavaScript', 'Julia', 'Kotlin', 'LISP', 'Matlab', 'Node.js', 'Objective-C', 'PHP', 'Perl', 'PowerShell', 'Python', 'R', 'Ruby', 'Rust', 'SQL', 'Scala', 'Swift', 'TypeScript', 'VBA'

X_sample = pd.DataFrame(X_sample.reshape(1,-1))
X_sample.columns = X_test.columns

y_pred = pipe.predict(X_sample); y_pred[0].round(2)

# ---------------------------------------------------------------------------------------------------------------
# MODEL Lasso regression ----------------------------------------------------------------------------------------

import timeit
start = timeit.default_timer()

from sklearn.linear_model import Lasso
model = Lasso()

from sklearn.pipeline import make_pipeline
pipe = make_pipeline(column_trans, scaler, model)
pipe.fit(X_train, y_train)

y_pred = pipe.predict(X_test)

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
outcome = pd.DataFrame({'y_test':y_test, 'y_pred':y_pred})
outcome['difference'] = outcome['y_test'] - outcome['y_pred']
outcome['difference_percentage'] = round(outcome.difference/(outcome.y_test/100),6)

print('Lasso:')
print('PROC: ', round(outcome.difference_percentage.abs().mean(),2),'%')
print('MAE: ', round(mean_absolute_error(y_test, y_pred),4))
print('RMSE: ', round(np.sqrt(mean_squared_error(y_test, y_pred)),4))
print('R2:', round(r2_score(y_test, y_pred),4))

stop = timeit.default_timer()
print('Time: ', stop - start)

# ---------------------------------------------------------------------------------------------------------------
# MODEL Ridge regression ----------------------------------------------------------------------------------------

import timeit
start = timeit.default_timer()

from sklearn.linear_model import Ridge
model = Ridge()

from sklearn.pipeline import make_pipeline
pipe = make_pipeline(column_trans, scaler, model)
pipe.fit(X_train, y_train)

y_pred = pipe.predict(X_test)

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
outcome = pd.DataFrame({'y_test':y_test, 'y_pred':y_pred})
outcome['difference'] = outcome['y_test'] - outcome['y_pred']
outcome['difference_percentage'] = round(outcome.difference/(outcome.y_test/100),6)

print('Ridge:')
print('PROC: ', round(outcome.difference_percentage.abs().mean(),2),'%')
print('MAE: ', round(mean_absolute_error(y_test, y_pred),4))
print('RMSE: ', round(np.sqrt(mean_squared_error(y_test, y_pred)),4))
print('R2:', round(r2_score(y_test, y_pred),4))

stop = timeit.default_timer()
print('Time: ', stop - start)

# ---------------------------------------------------------------------------------------------------------------
# MODEL Decision tree regression --------------------------------------------------------------------------------

def DecisionTree():
    import timeit
    start = timeit.default_timer()

    from sklearn.tree import DecisionTreeRegressor
    model = DecisionTreeRegressor(random_state=0)

    from sklearn.pipeline import make_pipeline
    pipe = make_pipeline(column_trans, scaler, model)
    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)

    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    outcome = pd.DataFrame({'y_test':y_test, 'y_pred':y_pred})
    outcome['difference'] = outcome['y_test'] - outcome['y_pred']
    outcome['difference_percentage'] = round(outcome.difference/(outcome.y_test/100),6)

    print('DecisionTree:')
    print('PROC: ', round(outcome.difference_percentage.abs().mean(),2),'%')
    print('MAE: ', round(mean_absolute_error(y_test, y_pred),4))
    print('RMSE: ', round(np.sqrt(mean_squared_error(y_test, y_pred)),4))
    print('R2:', round(r2_score(y_test, y_pred),4))

    stop = timeit.default_timer()
    print('Time: ', stop - start)

    return pipe

# Sample
X_sample = np.array(["Czech Republic", "Master’s degree", "2", "10,000 or more employees", "MacOS", "25-34",
                      "0",       "0",         "1",     "0",  "0",  "0",    "0",      "0",       "0",      "0",     "0",      "0",      "0",    "0",  "0",    "0",       "1",       "0",      "0",       "0",       "0",     "0",     "0",      "0",      "0",         "0",       "0",    "0",       "0",        "1",   "1",  "0",    "0",    "1",    "0",    "0",        "0",       "0"])
                    # 'APL', 'Assembly', 'Bash/Shell', 'C', 'C#', 'C++', 'COBOL', 'Clojure', 'Crystal', 'Dart', 'Delphi', 'Elixir', 'Erlang', 'F#', 'Go', 'Groovy', 'HTML/CSS', 'Haskell', 'Java', 'JavaScript', 'Julia', 'Kotlin', 'LISP', 'Matlab', 'Node.js', 'Objective-C', 'PHP', 'Perl', 'PowerShell', 'Python', 'R', 'Ruby', 'Rust', 'SQL', 'Scala', 'Swift', 'TypeScript', 'VBA'

X_sample = pd.DataFrame(X_sample.reshape(1,-1))
X_sample.columns = X_test.columns

y_pred = DecisionTree().predict(X_sample); y_pred[0].round(2)

# ---------------------------------------------------------------------------------------------------------------
# MODEL Random forest regression --------------------------------------------------------------------------------

import timeit
start = timeit.default_timer()

from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(n_estimators=500, random_state=0)

from sklearn.pipeline import make_pipeline
pipe = make_pipeline(column_trans, scaler, model)
pipe.fit(X_train, y_train)

y_pred = pipe.predict(X_test)

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
outcome = pd.DataFrame({'y_test':y_test, 'y_pred':y_pred})
outcome['difference'] = outcome['y_test'] - outcome['y_pred']
outcome['difference_percentage'] = round(outcome.difference/(outcome.y_test/100),6)

print('RandomForest:')
print('PROC: ', round(outcome.difference_percentage.abs().mean(),2),'%')
print('MAE: ', round(mean_absolute_error(y_test, y_pred),4))
print('RMSE: ', round(np.sqrt(mean_squared_error(y_test, y_pred)),4))
print('R2:', round(r2_score(y_test, y_pred),4))

stop = timeit.default_timer()
print('Time: ', stop - start)

# ---------------------------------------------------------------------------------------------------------------
# Modeling boosting/tunning based -------------------------------------------------------------------------------

# Encoding
from sklearn.preprocessing import LabelEncoder
le_country = LabelEncoder()
df['Country'] = le_country.fit_transform(df['Country'])
df['Country'].unique()

from sklearn.preprocessing import LabelEncoder
le_education = LabelEncoder()
df['EdLevel'] = le_education.fit_transform(df['EdLevel'])
df["EdLevel"].unique()

from sklearn.preprocessing import LabelEncoder
le_organization = LabelEncoder()
df['OrgSize'] = le_organization.fit_transform(df['OrgSize'])
df["OrgSize"].unique()

from sklearn.preprocessing import LabelEncoder
le_system = LabelEncoder()
df['OpSys'] = le_system.fit_transform(df['OpSys'])
df["OpSys"].unique()

from sklearn.preprocessing import LabelEncoder
le_age = LabelEncoder()
df['Age'] = le_age.fit_transform(df['Age'])
df["Age"].unique()

# Train/Test split
X = df.drop(columns=['Salary'], axis=1)
y = df['Salary']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1001)

# ---------------------------------------------------------------------------------------------------------------
# MODEL XGBoost -------------------------------------------------------------------------------------------------

# Feature names (needed because of the .values)
feature_columns = list(df.columns.values)
feature_columns = feature_columns[1:]; feature_columns

# Transform
import xgboost as xgb
Train = xgb.DMatrix(X_train, label = y_train, feature_names = feature_columns)
Test = xgb.DMatrix(X_test, label = y_test, feature_names = feature_columns)

# Parameters
parameters = {'learning_rate' : 0.05, # ETA learning rate
              'max_depth' : 20, # bigger tree more details
              'colsample_bytree' : 1, # how much of the tree should be analysed
              'subsample' : 1, # share of observation in each tree
              'min_child_weight' : 1, # weights of each observation
              'gamma' : 1, # how fast should the tree be split
              'eval_metric' : "rmse", # evaluation
              'objective' : "reg:squarederror"} # squared order

# Model
model = xgb.train(params=parameters, 
                  dtrain=Train,
                  num_boost_round = 80)

# Prediction
y_pred = model.predict(Test); y_pred

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

outcome = pd.DataFrame({'y_test':y_test, 'y_pred':y_pred})
outcome['difference'] = outcome['y_test'] - outcome['y_pred']
outcome['difference_percentage'] = round(outcome.difference/(outcome.y_test/100),6)

print('XGBoost:')
print('PROC: ', round(outcome.difference_percentage.abs().mean(),2),'%')
print('MAE: ', round(mean_absolute_error(y_test, y_pred),4))
print('RMSE: ', round(np.sqrt(mean_squared_error(y_test, y_pred)),4))
print('R2:', round(r2_score(y_test, y_pred),4))

# Prediction comparison visualisation
def viz():
    sn.distplot(y_test, label='Actual Result', hist=False, color='red')
    sn.distplot(y_pred, label='Predicted Result', hist=False, color='blue')
    plt.legend();

viz()

# Plot importance
xgb.plot_importance(model, max_num_features=15);

# Sample
X_sample = np.array([["Czech Republic", "Master’s degree", "2", "10,000 or more employees", "MacOS", "25-34",
                      "0",       "0",         "1",     "0",  "0",  "0",    "0",      "0",       "0",      "0",     "0",      "0",      "0",    "0",  "0",    "0",       "1",       "0",      "0",       "0",       "0",     "0",     "0",      "0",      "0",         "0",       "0",    "0",       "0",        "1",   "1",  "0",    "0",    "1",    "0",    "0",        "0",       "0"]])
                    # 'APL', 'Assembly', 'Bash/Shell', 'C', 'C#', 'C++', 'COBOL', 'Clojure', 'Crystal', 'Dart', 'Delphi', 'Elixir', 'Erlang', 'F#', 'Go', 'Groovy', 'HTML/CSS', 'Haskell', 'Java', 'JavaScript', 'Julia', 'Kotlin', 'LISP', 'Matlab', 'Node.js', 'Objective-C', 'PHP', 'Perl', 'PowerShell', 'Python', 'R', 'Ruby', 'Rust', 'SQL', 'Scala', 'Swift', 'TypeScript', 'VBA'

X_sample[:,0] = le_country.transform(X_sample[:,0])
X_sample[:,1] = le_education.transform(X_sample[:,1])
X_sample[:,3] = le_organization.transform(X_sample[:,3])
X_sample[:,4] = le_system.transform(X_sample[:,4])
X_sample[:,5] = le_age.transform(X_sample[:,5])
X_sample = X_sample.astype(float)

X_sample = xgb.DMatrix(X_sample, feature_names = feature_columns)

y_pred = model.predict(X_sample); y_pred[0]

# ---------------------------------------------------------------------------------------------------------------
# MODEL LightGBM ------------------------------------------------------------------------------------------------

# Transform
import lightgbm as lgb
Train = lgb.Dataset(X_train, y_train)
Test = lgb.Dataset(X_test, y_test, reference=Train)

# Defining parameters 
parameters = {
    'task': 'train', 
    'boosting': 'gbdt',
    'objective': 'regression',
    'num_leaves': 10,
    'learnnig_rage': 0.05,
    'metric': {'l2','l1'},
    'verbose': -1
}

# Model
model = lgb.train(parameters,
                 train_set=Train,
                 valid_sets=Test,
                 early_stopping_rounds=80)

# Prediction
y_pred = model.predict(X_test); y_pred

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

outcome = pd.DataFrame({'y_test':y_test, 'y_pred':y_pred})
outcome['difference'] = outcome['y_test'] - outcome['y_pred']
outcome['difference_percentage'] = round(outcome.difference/(outcome.y_test/100),6)

print('LightGBM:')
print('PROC: ', round(outcome.difference_percentage.abs().mean(),2),'%')
print('MAE: ', round(mean_absolute_error(y_test, y_pred),4))
print('RMSE: ', round(np.sqrt(mean_squared_error(y_test, y_pred)),4))
print('R2:', round(r2_score(y_test, y_pred),4))

# Prediction comparison visualisation
def viz():
    sn.distplot(y_test, label='Actual Result', hist=False, color='red')
    sn.distplot(y_pred, label='Predicted Result', hist=False, color='blue')
    plt.legend();

viz()

# Sample
X_sample = np.array([["Czech Republic", "Master’s degree", "2", "10,000 or more employees", "MacOS", "25-34",
                      "0",       "0",         "1",     "0",  "0",  "0",    "0",      "0",       "0",      "0",     "0",      "0",      "0",    "0",  "0",    "0",       "1",       "0",      "0",       "0",       "0",     "0",     "0",      "0",      "0",         "0",       "0",    "0",       "0",        "1",   "1",  "0",    "0",    "1",    "0",    "0",        "0",       "0"]])
                    # 'APL', 'Assembly', 'Bash/Shell', 'C', 'C#', 'C++', 'COBOL', 'Clojure', 'Crystal', 'Dart', 'Delphi', 'Elixir', 'Erlang', 'F#', 'Go', 'Groovy', 'HTML/CSS', 'Haskell', 'Java', 'JavaScript', 'Julia', 'Kotlin', 'LISP', 'Matlab', 'Node.js', 'Objective-C', 'PHP', 'Perl', 'PowerShell', 'Python', 'R', 'Ruby', 'Rust', 'SQL', 'Scala', 'Swift', 'TypeScript', 'VBA'

X_sample[:,0] = le_country.transform(X_sample[:,0])
X_sample[:,1] = le_education.transform(X_sample[:,1])
X_sample[:,3] = le_organization.transform(X_sample[:,3])
X_sample[:,4] = le_system.transform(X_sample[:,4])
X_sample[:,5] = le_age.transform(X_sample[:,5])
X_sample = X_sample.astype(float)

y_pred = model.predict(X_sample); y_pred[0].round(2)

# TESTING
df = pd.DataFrame({'Salary' : [23000,50000,32000,30000], 
                   "Country" : ['Sweden','Slovakia','Austria','Turkey'],
                   "EdLevel" : ['Master’s degree','Master’s degree','Master’s degree','Bachelor’s degree'], 
                   "Age1stCode" : [5,30,15,18], 
                   "LanguageHaveWorkedWith" : ['C;Python','Ruby;SQL','SQL','R;Python;SQL'], 
                   })
df

df = pd.concat([df, df['LanguageHaveWorkedWith'].str.get_dummies(sep=';')], axis=1); df

# DESIRED
df = pd.DataFrame({'Salary' : [23000,50000,32000,30000], 
                   "Country" : ['Sweden','Slovakia','Austria','Turkey'],
                   "EdLevel" : ['Master’s degree','Master’s degree','Master’s degree','Bachelor’s degree'], 
                   "Age1stCode" : [5,30,15,18], 
                   "LanguageHaveWorkedWith" : ['C;Python','SQL','SQL','R;Python;SQL'],
                   "C" : [1,0,0,0],
                   "Python" : [1,0,0,1],
                   "SQL" : [0,1,1,1],
                   "R" : [0,0,0,1]
                   })
df