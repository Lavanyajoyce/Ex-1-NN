<H3>ENTER YOUR NAME : THILAGAVATHI S</H3>
<H3>ENTER YOUR REGISTER NO.: 212222220053</H3>
<H3>EX. NO.1</H3>
<H3>DATE: 11-03-2024 </H3>
<H1 ALIGN =CENTER> Introduction to Kaggle and Data preprocessing</H1>

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
import pandas as pd                                                 # Importing Libraries
import io
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
df=pd.read_csv("Churn_Modelling.csv",index_col="RowNumber")         # Read the dataset from drive
df.head()
df.isnull().sum()                                                   # Finding Missing Values
df.duplicated().sum()                                               # Check For Duplicates
df=df.drop(['Surname', 'Geography','Gender'], axis=1)               # Remove Unnecessary Columns
scaler=StandardScaler()                                             # Normalize the dataset
df=pd.DataFrame(scaler.fit_transform(df))
df.head()
X,Y=df.iloc[:,:-1].values ,df.iloc[:,-1].values                     # Split the dataset into input and output
print('Input:\n',X,'\nOutput:\n',Y) 
Xtrain,Xtest,Ytrain,Ytest = train_test_split(X, Y, test_size=0.2)   # Splitting the data for training & Testing
print("Xtrain:\n" ,Xtrain, "\nXtest:\n", Xtest)                     # X Train and Test
print("\nYtrain:\n" ,Ytrain, "\nYtest:\n", Ytest)                   # Y Train and Test


print("Lenght of X_test ",len(Xtest))
```

## OUTPUT:
### Dataset:
![image](https://github.com/poojaanbu0/Ex-1-NN/assets/119390329/a7af9530-258a-46f6-9076-bc03473b87bd)

### Null values:
![image](https://github.com/poojaanbu0/Ex-1-NN/assets/119390329/ebd54c00-2d18-4768-b2bf-25b8ab4af5ad)

### Normalised data:
![image](https://github.com/poojaanbu0/Ex-1-NN/assets/119390329/ba903eca-dd96-4106-80b5-8ceab9344312)

### Data splitting:
![image](https://github.com/poojaanbu0/Ex-1-NN/assets/119390329/f0e89ce0-8674-4008-9208-13ab68e4a5f1)

### Train and Test data:
![image](https://github.com/poojaanbu0/Ex-1-NN/assets/119390329/e9eff234-81da-45f1-a103-0f93c043954b)

### Length:
![image](https://github.com/poojaanbu0/Ex-1-NN/assets/119390329/7420e8f0-8172-47d5-ab7c-b7b52a8a6f42)


## RESULT:
Thus, Implementation of Data Preprocessing is done in python  using a data set downloaded from Kaggle.


