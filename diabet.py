#Feature Engineering

#Veri setindeki değişkenler

# Pregnancies: Hamilelik sayısı
# Glucose Oral: glikoz tolerans testinde 2 saatlik plazma glikoz konsantrasyonu
# Blood Pressure: Kan Basıncı (Küçük tansiyon) (mm Hg)
# SkinThickness: Cilt Kalınlığı
# Insulin: 2 saatlik serum insülini (mu U/ml)
# DiabetesPedigreeFunction: Fonksiyon (Oral glikoz tolerans testinde 2 saatlik plazma glikoz konsantrasyonu)
# BMI: Vücut kitle endeksi
# Age: Yaş (yıl)
# Outcome: Hastalığa sahip (1) ya da değil (0)

#outcome:hedef değişken
#1:diyabet test sonucu pozitif
#0:iyabet test sonucu negatif

#Kütüphaneler
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier

pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 600)

#Loading the dataset
def load_diabet():
    data = pd.read_csv("HAFTA_06/1-Notes/Ödevler/diabetes.csv")
    return data

df_= load_diabet()
df=df_.copy()
df.head()


#Dataset analysis
df.shape
df.info()
df.describe([0, 0.05,0.25, 0.50, 0.75,0.95, 0.99, 1]).T


#Average of numerical variables relative to the target variable

df.groupby("Outcome").agg("mean")


# Outlier observation analysis

#Threshold determination
def outlier_thresholds(dataframe, col_name, q1=0.25, q3=0.75):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

low, up = outlier_thresholds(df, "Pregnancies")


#We checked for outliers
def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

check_outlier(df,"Pregnancies")
check_outlier(df,"BloodPressure")
check_outlier(df,"SkinThickness")
check_outlier(df,"Insulin")
check_outlier(df,"BMI")
check_outlier(df,"DiabetesPedigreeFunction")
check_outlier(df,"Age")


#Multivariate Outlier Analysis: Local Outlier Factor
clf = LocalOutlierFactor(n_neighbors=20)
clf.fit_predict(df)

df_scores = clf.negative_outlier_factor_
np.sort(df_scores)[0:5]

scores = pd.DataFrame(np.sort(df_scores))
scores.plot(stacked=True, xlim=[0, 50], style='.-')
plt.show()

th = np.sort(df_scores)[10]
df[df_scores < th]
df[df_scores < th].shape


#Missing observation analysis
df.isnull().sum()

#There are no missing observations in the data. But some variables have a minimum value of zero.
#For example, a person's skin thickness cannot be zero..
#We should also take the zero values as missing observations.

#We replaced the zero values in the variables other than "Pregnancies" and "Outcome" with "NAN".
zero_columns = [i for i in df.columns if (df[i].min() == 0 and i not in ["Pregnancies", "Outcome"])]

for i in zero_columns:
    df[[i]] = df[[i]].replace(0, np.NaN)

#We checked how many missing values.
df.isnull().sum()


#Solving missing value problem with KNN

scaler = MinMaxScaler()
df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
df.head()


imputer = KNNImputer(n_neighbors=5)
df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
df = pd.DataFrame(scaler.inverse_transform(df), columns=df.columns)
df.head()

#We checked again to see if there are any missing observations

df.isnull().sum()

#Solving the outlier problem: Re-assignment with thresholds
def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

replace_with_thresholds(df,"Pregnancies")
replace_with_thresholds(df,"BloodPressure")
replace_with_thresholds(df,"SkinThickness")
replace_with_thresholds(df,"Insulin")
replace_with_thresholds(df,"BMI")
replace_with_thresholds(df,"DiabetesPedigreeFunction")
replace_with_thresholds(df,"Age")

#We checked again for outliers

check_outlier(df,"Pregnancies")
check_outlier(df,"BloodPressure")
check_outlier(df,"SkinThickness")
check_outlier(df,"Insulin")
check_outlier(df,"BMI")
check_outlier(df,"DiabetesPedigreeFunction")
check_outlier(df,"Age")


#Correlation analysis
df.corrwith(df["Outcome"]).sort_values(ascending=False)
corr_df = df.corr()

sns.heatmap(corr_df, annot=True, xticklabels=corr_df.columns, yticklabels=corr_df.columns)
plt.show()

#Feature Engineering


#Generating new variables according to the Corr relationship

df["Age-Insul"] = df["Age"] * df["Insulin"]
df["Age-BMI"] = df["Age"] * df["BMI"]
df["Preg-Insul"] = df["Pregnancies"] * df["Insulin"]
df["Gluc-Insul"] = df["Glucose"] * df["Insulin"]
df["SkinT-Age"] = df["SkinThickness"] * df["Age"]
df["Preg-SkinT"] = df["SkinThickness"] * df["Pregnancies"]

#We converted some numeric variables to categorical variables.

#Classification of blood pressure variable

def new_insulin(row):
    if row["BloodPressure"]>80:
        return "Hipertansiyon"
    elif row["BloodPressure"]<60:
        return "Hipotansiyon"
    else:
        return "Normal"

df = df.assign(NewInsulin=df.apply(new_insulin, axis=1))

#Classifying the age variable

def new_age(row):
    if row["Age"]>40:
        return "Olgunyaş"
    elif row["Age"]<=40 and row["Age"]>30:
        return "Ortayaş"
    elif row["Age"]<=30 and row["Age"]>=25:
        return "Gençyaş"
    else:
        return "Ergen"

df = df.assign(NewAge=df.apply(new_age, axis=1))



#Classifying the BMI variable

def new_bmı(row):
    if row["BMI"]>40:
        return "Morbidobez"
    elif row["BMI"]<=40 and row["BMI"]>35:
        return "Tip2obez"
    elif row["BMI"]<=35 and row["BMI"]>30:
        return "Tip1obez"
    elif row["BMI"]<=30 and row["BMI"]>25:
        return "Fazlakilolu"
    else:
        return "Normal"

df = df.assign(NewBMI=df.apply(new_bmı, axis=1))



#Classifying the glucose variable

def new_glucose(row):
    if row["Glucose"]>200:
        return "Riskli"
    elif row["Glucose"] <=200 and row["Glucose"] >140:
        return "Diabet"
    else:
        return "Normal"

df = df.assign(NewGlucose=df.apply(new_glucose, axis=1))

df.head()

#By converting categorical variables to 1-0, we have expressed it in a language that the machine learning algorithm can understand.
#One-Hot Encoding
df = pd.get_dummies(df, columns=["NewGlucose"], drop_first=True)
df = pd.get_dummies(df, columns=["NewBMI"], drop_first=True)
df = pd.get_dummies(df, columns=["NewAge"], drop_first=True)
df = pd.get_dummies(df, columns=["NewInsulin"], drop_first=True)


df.head()


#Model creation

#Identifying dependent and independent variables
y = df["Outcome"]
X = df.drop(["Outcome"], axis=1)

#Testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=17)

#Model-1
rf_model = DecisionTreeRegressor(random_state=46).fit(X_train, y_train)
y_pred = rf_model.predict(X_test)
accuracy_score(y_pred, y_test)

#Model-2
rf_model = RandomForestClassifier(random_state=46).fit(X_train, y_train)
y_pred = rf_model.predict(X_test)
accuracy_score(y_pred, y_test)

#Conclusion

#DecisionTreeRegressor - 0.7705
#RandomForestClassifier- 0.800

#Random Forest Classifier has higher prediction success, so this can be preferred.


