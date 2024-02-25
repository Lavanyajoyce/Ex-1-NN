### ENTER YOUR NAME: Hariharan M
### ENTER YOUR REGISTER NO: 212221220015
### EX. NO.1
### DATE: 25/02/2024
# Introduction to Kaggle and Data preprocessing

## AIM:

To perform Data preprocessing in a data set downloaded from Kaggle

## EQUIPMENTS REQUIRED:
Hardware – PCs
Anaconda – Python 3.7 Installation / Google Colab /Jupiter Notebook

## RELATED THEORETICAL CONCEPT:

**Kaggle :**
Kaggle, a subsidiary of Google LLC, is an online community of data scientists and machine learning practitioners. Kaggle allows users to find and publish data sets, explore and build models in a web-based data-science environment, work with other data scientists and machine learning engineers, and enter competitions to solve data science challenges.

**Data Preprocessing:**

Pre-processing refers to the transformations applied to our data before feeding it to the algorithm. Data Preprocessing is a technique that is used to convert the raw data into a clean data set. In other words, whenever the data is gathered from different sources it is collected in raw format which is not feasible for the analysis.
Data Preprocessing is the process of making data suitable for use while training a machine learning model. The dataset initially provided for training might not be in a ready-to-use state, for e.g. it might not be formatted properly, or may contain missing or null values.Solving all these problems using various methods is called Data Preprocessing, using a properly processed dataset while training will not only make life easier for you but also increase the efficiency and accuracy of your model.

**Need of Data Preprocessing :**

For achieving better results from the applied model in Machine Learning projects the format of the data has to be in a proper manner. Some specified Machine Learning model needs information in a specified format, for example, Random Forest algorithm does not support null values, therefore to execute random forest algorithm null values have to be managed from the original raw data set.
Another aspect is that the data set should be formatted in such a way that more than one Machine Learning and Deep Learning algorithm are executed in one data set, and best out of them is chosen.


## ALGORITHM:
STEP 1:Importing the libraries<BR>
STEP 2:Importing the dataset<BR>
STEP 3:Taking care of missing data<BR>
STEP 4:Encoding categorical data<BR>
STEP 5:Normalizing the data<BR>
STEP 6:Splitting the data into test and train<BR>

##  PROGRAM:
```
import pandas as pd
import io
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

data = pd.read_csv("Churn_Modelling.csv")
data
data.head()

X=data.iloc[:,:-1].values
X

y=data.iloc[:,-1].values
y

data.isnull().sum()

data.duplicated()

data.describe()

data = data.drop(['Surname', 'Geography','Gender'], axis=1)
data.head()

scaler=MinMaxScaler()
df1=pd.DataFrame(scaler.fit_transform(data))
print(df1)

X_train ,X_test ,y_train,y_test=train_test_split(X,y,test_size=0.2)

X_train

X_test

print("Lenght of X_test ",len(X_test))

```

## OUTPUT:
## Data set:
![image](https://github.com/Lavanyajoyce/Ex-1-NN/assets/116946289/b105b1b8-fc0e-4721-abf6-edf4ee879a40)

## X Values:
![image](https://github.com/Lavanyajoyce/Ex-1-NN/assets/116946289/06f68ffb-aa19-43ba-9219-6327af4bd8f7)

## Y Values:
![image](https://github.com/Lavanyajoyce/Ex-1-NN/assets/116946289/79dfa3b6-c446-4eee-9c2a-f7c181d4204a)

## Null Values:
![image](https://github.com/Lavanyajoyce/Ex-1-NN/assets/116946289/44674e69-cb65-476d-930f-b09891c0a7ef)

## Duplicated Values:
![image](https://github.com/Lavanyajoyce/Ex-1-NN/assets/116946289/919984f3-0937-4b07-b085-bbf6a202fd33)

## Description:
![image](https://github.com/Lavanyajoyce/Ex-1-NN/assets/116946289/da26d30b-9300-4fa5-9ba2-3d5e8fda23d4)

## Normalized Dataset:
![image](https://github.com/Lavanyajoyce/Ex-1-NN/assets/116946289/721e53c9-9dcc-4f9d-adb1-a82faf7b0339)

## Training Data:
![image](https://github.com/Lavanyajoyce/Ex-1-NN/assets/116946289/ff6892d2-b038-407c-857d-98b5271ff908)

## Testing Data:
![image](https://github.com/Lavanyajoyce/Ex-1-NN/assets/116946289/ceccb457-ef5d-41ad-a02b-7d6ece049428)

## RESULT:
Thus, Implementation of Data Preprocessing is done in python  using a data set downloaded from Kaggle.


