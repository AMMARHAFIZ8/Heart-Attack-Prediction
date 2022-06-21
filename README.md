# Assignment for Heart Attack Prediction using Heart.csv dataset.

### EDA 
First step Data Loading





Boxplot

![Alt text](https://github.com/AMMARHAFIZ8/heart_assignment/blob/main/Figure%202022-06-21%20192108%20boxplot.png)

sex

![Alt text](https://github.com/AMMARHAFIZ8/heart_assignment/blob/main/Figure%202022-06-21%20192035%20sex.png)

age

![Alt text](https://github.com/AMMARHAFIZ8/heart_assignment/blob/main/Figure%202022-06-21%20192256%20age.png)

trtbps

![Alt text](https://github.com/AMMARHAFIZ8/heart_assignment/blob/main/Figure%202022-06-21%20192229%20trtbps.png)

chol

![Alt text](https://github.com/AMMARHAFIZ8/heart_assignment/blob/main/Figure%202022-06-21%20192210%20chol.png)

oldpeak

![Alt text](https://github.com/AMMARHAFIZ8/heart_assignment/blob/main/Figure%202022-06-21%20192130%20oldpeak.png)

thalachh

![Alt text](https://github.com/AMMARHAFIZ8/heart_assignment/blob/main/Figure%202022-06-21%20192148%20thalachh.png)


### categorical = 'sex', 'cp', 'fbs', 'restecg', 'exng', 'slp', 'caa', 'thall', 'output'.
### continuous = 'age', 'trtbps', 'chol', 'thalachh', 'oldpeak'.

## Discussion

#### The best pipeline for this heart dataset is Pipeline(steps=[('mmsscaler', MinMaxScaler()),('rf', RandomForestClassifier())]) with accuracy of 0.7802197802197802
#### With the accuracy score of 0.7472527472527473


#### Model Deployment 
![Alt text](https://github.com/AMMARHAFIZ8/heart_assignment/blob/main/Deploy%20App%20for%20Heart%20Attack%20(Browser).PNG)




[Credit  ](http://archive.ics.uci.edu/ml/datasets/Heart+Disease)
