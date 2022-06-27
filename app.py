# FLASK

# conda activate enviro
# cd "/Users/HP/OneDrive/Documents/Python Anaconda/Flask_Salary_App/Salary_2021"
# MAC: export FLASK_APP=app.py WINDOWS: set FLASK_APP=app.py
# flask run

from flask import Flask, render_template, request
import numpy as np
import pandas as pd
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 100)

app = Flask(__name__)

@app.route('/')
def index():

    # Importing
    import os
    path = "/Users/HP/OneDrive/Documents/Python Anaconda/Flask_Salary_App/Salary_2021"
    os.chdir(path)
    os.listdir()

    df = pd.read_csv('survey_clean.csv')

    # PUSH
    countries = sorted(df['Country'].unique())
    educations = ['Less than a Bachelor', 'Bachelor’s degree', 'Master’s degree', 'Post grad']
    years = [str(x) for x in range(50 + 1)]
    organizations = ['2 to 9 employees', '10 to 19 employees', '20 to 99 employees', '100 to 499 employees', '500 to 999 employees', '1,000 to 4,999 employees', '5,000 to 9,999 employees', '10,000 or more employees', 'Just me - I am a freelancer, sole proprietor, etc.', 'I don’t know']
    systems = ['Windows', 'Linux', 'MacOS', 'Other']
    ages = ['Under 18', '18-24', '25-34', '35-44', '45-54', '55-64', '65+', 'Do not want to say!']

    # Visualisation countries
    import json
    import plotly
    import plotly.express as px
    fig1 = px.box(df.sort_values('Country'), x="Country", y="Salary", 
                                    template='simple_white',
                                    title=f"<b>Global</b> - Salary based on country (USD)")
    fig1.update_xaxes(tickangle=90, tickmode = 'array', tickvals = df.sort_values('Country')['Country'])
    
    graphJSON1 = json.dumps(fig1, cls=plotly.utils.PlotlyJSONEncoder)

    return render_template('index.html', countries=countries, educations=educations, years=years, organizations=organizations, systems=systems, ages=ages, graphJSON1=graphJSON1)

@app.route('/predict', methods=['POST'])
def predict():

    # GET
    Country = request.form.get('country')
    EdLevel = request.form.get('education')
    YearsCodePro = request.form.get('year')
    OrgSize = request.form.get('organization')
    OpSys = request.form.get('system')
    Age = request.form.get('age')
    
    APL = request.form.get('APL')
    Assembly = request.form.get('Assembly')
    Bash_Shell = request.form.get('Bash_Shell')
    C = request.form.get('C')
    C_sharp = request.form.get('C_sharp')
    C_plus_plus = request.form.get('C_plus_plus')
    COBOL = request.form.get('COBOL')
    Clojure = request.form.get('Clojure')
    Crystal = request.form.get('Crystal')
    Dart = request.form.get('Dart')
    Delphi = request.form.get('Delphi')
    Elixir = request.form.get('Elixir')
    Erlang = request.form.get('Erlang')
    F_sharp = request.form.get('F_sharp')
    Go = request.form.get('Go')
    Groovy = request.form.get('Groovy')
    HTML_CSS = request.form.get('HTML_CSS')
    Haskell = request.form.get('Haskell')
    Java = request.form.get('Java')
    JavaScript = request.form.get('JavaScript')
    Julia = request.form.get('Julia')
    Kotlin = request.form.get('Kotlin')
    LISP = request.form.get('LISP')
    Matlab = request.form.get('Matlab')
    Node_js = request.form.get('Node_js')
    Objective_C = request.form.get('Objective_C')
    PHP = request.form.get('PHP')
    Perl = request.form.get('Perl')
    PowerShell = request.form.get('PowerShell')
    Python = request.form.get('Python')
    R = request.form.get('R')
    Ruby = request.form.get('Ruby')
    Rust = request.form.get('Rust')
    SQL = request.form.get('SQL')
    Scala = request.form.get('Scala')
    Swift = request.form.get('Swift')
    TypeScript = request.form.get('TypeScript')
    VBA = request.form.get('VBA')

    print(Country, EdLevel, YearsCodePro, OrgSize, OpSys, Age, APL, VBA)
    print(type(Country))
    print(type(YearsCodePro))
    print(type(APL))

    # Importing
    import os
    path = "/Users/HP/OneDrive/Documents/Python Anaconda/Flask_Salary_App/Salary_2021"
    os.chdir(path)
    os.listdir()

    df = pd.read_csv('survey_clean.csv')

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

    # Linear regression
    from sklearn.linear_model import LinearRegression # CHANGE
    model = LinearRegression(normalize=True) # CHANGE

    from sklearn.pipeline import make_pipeline
    pipe = make_pipeline(column_trans, scaler, model)
    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)

    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    outcome = pd.DataFrame({'y_test':y_test, 'y_pred':y_pred})
    outcome['difference'] = outcome['y_test'] - outcome['y_pred']
    outcome['difference_percentage'] = round(outcome.difference/(outcome.y_test/100),6)

    MAE = round(mean_absolute_error(y_test, y_pred),4)
    RMSE = round(np.sqrt(mean_squared_error(y_test, y_pred)),4)
    R2 = round(r2_score(y_test, y_pred),4)

    # Input
    input = pd.DataFrame([[Country, EdLevel, YearsCodePro, OrgSize, OpSys, Age, APL, Assembly, Bash_Shell, C, C_sharp, C_plus_plus, COBOL, Clojure, Crystal, Dart, Delphi, Elixir, Erlang, F_sharp,  Go, Groovy, HTML_CSS, Haskell, Java, JavaScript, Julia, Kotlin, LISP, Matlab, Node_js, Objective_C, PHP, Perl, PowerShell, Python, R,  Ruby, Rust, SQL, Scala, Swift, TypeScript, VBA]], columns=['Country', 'EdLevel', 'YearsCodePro', 'OrgSize', 'OpSys', 'Age', 'APL', 'Assembly', 'Bash/Shell', 'C', 'C#', 'C++', 'COBOL', 'Clojure', 'Crystal', 'Dart', 'Delphi', 'Elixir', 'Erlang', 'F#', 'Go', 'Groovy', 'HTML/CSS', 'Haskell', 'Java', 'JavaScript', 'Julia', 'Kotlin', 'LISP', 'Matlab', 'Node.js', 'Objective-C', 'PHP', 'Perl', 'PowerShell', 'Python', 'R', 'Ruby', 'Rust', 'SQL', 'Scala', 'Swift', 'TypeScript', 'VBA'])
    input.YearsCodePro = input.YearsCodePro.astype(np.float64)
    input.iloc[:, 6:] = input.iloc[:, 6:].astype(int)

    prediction = round(pipe.predict(input)[0],2)

    return {'prediction': prediction, 'MAE': MAE, 'RMSE': RMSE, 'R2': R2}

if __name__ == "__main__":
    app.run(debug=True, port=5001)

