from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectKBest
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import streamlit as st
# from warnings import filterwarnings
# filterwarnings('ignore')

st.title('Chronic Kidney Disease Detection')
st.write("""
Detect if someone has chronic kidney disease(CKD) using machine learning and python.
""")

df = pd.read_csv(
    'C:/Users/Obaidur Rahman/OneDrive/Desktop/chronic-kidney-disease-prediction/kidney_disease.csv')

columns = pd.read_csv(
    'C:/Users/Obaidur Rahman/OneDrive/Desktop/chronic-kidney-disease-prediction/data_description.txt', sep='-')
columns = columns.reset_index()
columns.columns = ['cols', 'abb_col_names']

df.columns = columns['abb_col_names'].values

# st.header('Data information:')
# st.dataframe(df)
# st.subheader('Data description:')
# st.write(df.describe())
# chart = st.bar_chart(df)

features = ['red_blood_cell_count',
            'packed_cell_volume', 'white_blood_cell_count']


def convert_dtype(df, feature):
    df[feature] = pd.to_numeric(df[feature], errors='coerce')


for feature in features:
    convert_dtype(df, feature)

df.drop(["id"], axis=1, inplace=True)


def extract_cat_num(df):
    cat_col = [col for col in df.columns if df[col].dtype == 'object']
    num_col = [col for col in df.columns if df[col].dtype != 'object']
    return cat_col, num_col


cat_col, num_col = extract_cat_num(df)

df['diabetes_mellitus'].replace(
    to_replace={'\tno': 'no', '\tyes': 'yes', ' yes': 'yes'}, inplace=True)
df['coronary_artery_disease'] = df['coronary_artery_disease'].replace(
    to_replace='\tno', value='no')
df['class'] = df['class'].replace(to_replace='ckd\t', value='ckd')

# st.subheader('Chart of Numerical features:')
# num_fig = plt.figure(figsize=(30, 20))
# for i, feature in enumerate(num_col):
#     plt.subplot(5, 3, i+1)
#     df[feature].hist()
#     plt.title(feature)
# st.pyplot(num_fig)

# st.subheader('Chart of Categorical features:')
# cat_fig = plt.figure(figsize=(20, 20))
# for i, feature in enumerate(cat_col):
#     plt.subplot(4, 3, i+1)
#     sns.countplot(df[feature])
# st.pyplot(cat_fig)

# st.subheader('Chart of CKD to Not CKD ratio:')
# ratio_fig = plt.figure(figsize=(5, 5))
# sns.countplot(x='class', data=df)
# st.pyplot(ratio_fig)

data = df.copy()

random_sample = data['red_blood_cells'].dropna().sample(
    data['red_blood_cells'].isnull().sum())

random_sample.index = data[data['red_blood_cells'].isnull()].index
data.loc[data['red_blood_cells'].isnull(), 'red_blood_cells'] = random_sample
# can be shown as after data cleaning


def Random_value_imputation(feature):
    random_sample = data[feature].dropna().sample(data[feature].isnull().sum())
    random_sample.index = data[data[feature].isnull()].index
    data.loc[data[feature].isnull(), feature] = random_sample


for col in num_col:
    Random_value_imputation(col)

Random_value_imputation('pus_cell')
mode = data['pus_cell_clumps'].mode()[0]


def impute_mode(feature):
    mode = data[feature].mode()[0]
    data[feature] = data[feature].fillna(mode)


for col in cat_col:
    impute_mode(col)

data[cat_col].isnull().sum()


# feature encoding

le = LabelEncoder()

for col in cat_col:
    data[col] = le.fit_transform(data[col])

ind_col = [col for col in data.columns if col != 'class']
dep_col = 'class'

X = data[ind_col]
Y = data[dep_col]

ordered_rank_features = SelectKBest(score_func=chi2, k=20)
ordered_feature = ordered_rank_features.fit(X, Y)

datascores = pd.DataFrame(ordered_feature.scores_, columns=["Score"])
dfcolumns = pd.DataFrame(X.columns)
features_rank = pd.concat([dfcolumns, datascores], axis=1)
features_rank.columns = ['Features', 'Score']

selected_columns = features_rank.nlargest(10, 'Score')['Features'].values

X_new = data[selected_columns]

X_train, X_test, Y_train, Y_test = train_test_split(
    X_new, Y, random_state=0, test_size=0.25)

# st.header('Selected features:')
# st.dataframe(X_new)
# st.subheader('Selected feature descrpition:')
# st.write(X_new.describe())

# get the feature input from users


def get_user_input():
    wbc_count1 = st.sidebar.number_input('wbc_count', value=7500.50)
    blood_glucose_random1 = st.sidebar.number_input(
        'blood_glucose_random', value=120.5925)
    blood_urea1 = st.sidebar.number_input('blood_urea', value=58.0755)
    serum_creatinine1 = st.sidebar.number_input('serum_creatinine', value=0.6)
    packed_cell_volume1 = st.sidebar.number_input(
        'packed_cell_volume', value=50)
    albumin1 = st.sidebar.number_input('albumin', value=4.0)
    haemoglobin1 = st.sidebar.number_input('haemoglobin', value=15.00)
    age1 = st.sidebar.number_input('age', value=40)
    hypertension1 = st.sidebar.select_slider(
        'hypertension', options=['No', 'Yes'])
    diabetes_mellitus1 = st.sidebar.select_slider(
        'diabetes_mellitus', options=['No', 'Yes'])

    if hypertension1 == 'No':
        hypertension1 = 0
    else:
        hypertension1 = 1

    if diabetes_mellitus1 == 'No':
        diabetes_mellitus1 = 0
    else:
        diabetes_mellitus1 = 1

    user_data = {
        'wbc_count': wbc_count1,
        'blood_glucose_random': blood_glucose_random1,
        'blood_urea': blood_urea1,
        'serum_creatinine': serum_creatinine1,
        'packed_cell_volume': packed_cell_volume1,
        'albumin': albumin1,
        'haemoglobin': haemoglobin1,
        'age': age1,
        'hypertension': hypertension1,
        'diabetes_mellitus': diabetes_mellitus1
    }

    features = pd.DataFrame(user_data, index=[0])
    return features


user_input = get_user_input()

st.subheader('User Input:')
st.write(user_input)

# building and training the model                                                                    

classifier = XGBClassifier()

params = {
    "learning_rate": [0.05, 0.20, 0.25],
    "max_depth": [5, 8, 10, 12],
    "min_child_weight": [1, 3, 5, 7],
    "gamma": [0.0, 0.1, 0.2, 0.4],
    "colsample_bytree": [0.3, 0.4, 0.7]
}

random_search = RandomizedSearchCV(
    classifier, param_distributions=params, n_iter=5, scoring='roc_auc', n_jobs=-1, cv=5, verbose=3)
random_search.fit(X_train, Y_train)
# random_search.best_estimator_
# random_search.best_params_
classifier = XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
                           colsample_bynode=1, colsample_bytree=0.3,
                           enable_categorical=False, gamma=0.0, gpu_id=-1,
                           importance_type=None, interaction_constraints='',
                           learning_rate=0.05, max_delta_step=0, max_depth=5,
                           min_child_weight=1, monotone_constraints='()',
                           n_estimators=100, n_jobs=12, num_parallel_tree=1,
                           predictor='auto', random_state=0, reg_alpha=0, reg_lambda=1,
                           scale_pos_weight=1, subsample=1, tree_method='exact',
                           validate_parameters=1, verbosity=None)

classifier.fit(X_train, Y_train)
Y_pred = classifier.predict(X_test)
# print(Y_pred)

# confusion = confusion_matrix(Y_test, Y_pred)
# print('Confusion Matrix:')
# print(confusion)
st.subheader('Accuracy of the model:')
accuracy = accuracy_score(Y_test, Y_pred)
st.write(f'{accuracy*100}%')


# user's data prediction
prediction = classifier.predict(user_input)


st.subheader('Classification:')
st.write(prediction)

diagnosis_pred = ''
if prediction[0] == 0:
    diagnosis_pred = 'Predicted diagnosis: **_CKD_**'
else:
    diagnosis_pred = 'Predicted diagnosis: **_No CKD_**'

st.subheader(diagnosis_pred)
