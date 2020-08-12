import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import classification_report, confusion_matrix, r2_score
from sklearn.metrics import f1_score
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer


#https://stackoverflow.com/questions/61171307/jupyter-notebook-shows-error-message-for-matplotlib-bad-key-text-kerning-factor

#Create your df here:
df= pd.read_csv("profiles.csv")
print(df.columns)
#Index(['age', 'body_type', 'diet', 'drinks', 'drugs', 'education', 'essay0',
#       'essay1', 'essay2', 'essay3', 'essay4', 'essay5', 'essay6', 'essay7',
#       'essay8', 'essay9', 'ethnicity', 'height', 'income', 'job',
#       'last_online', 'location', 'offspring', 'orientation', 'pets',
#       'religion', 'sex', 'sign', 'smokes', 'speaks', 'status'],
#      dtype='object')

#plt.hist(df.age, bins=20)
#plt.xlabel("Age")
#plt.ylabel("Frequency")
#plt.xlim(16, 80)
#plt.show()

#*******************************************************************************************************
#SIGN
print(df.sign.value_counts())
signs_mapping = {"gemini and it&rsquo;s fun to think about":0,
                 "scorpio and it&rsquo;s fun to think about":1,
                 "leo and it&rsquo;s fun to think about":2,
                 "libra and it&rsquo;s fun to think about":3,
                 "taurus and it&rsquo;s fun to think about":4,
                 "cancer and it&rsquo;s fun to think about":5,
                 "pisces and it&rsquo;s fun to think about":6,
                 "sagittarius and it&rsquo;s fun to think about":7,
                 "virgo and it&rsquo;s fun to think about":8,
                 "aries and it&rsquo;s fun to think about":9,
                 "aquarius and it&rsquo;s fun to think about":10,
                 "virgo but it doesn&rsquo;t matter":8,
                 "leo but it doesn&rsquo;t matter":2,
                 "cancer but it doesn&rsquo;t matter":5,
                 "gemini but it doesn&rsquo;t matter":0,
                 "taurus but it doesn&rsquo;t matter":4,
                 "libra but it doesn&rsquo;t matter":3,
                 "aquarius but it doesn&rsquo;t matter":10,
                 "capricorn and it&rsquo;s fun to think about":11,
                 "sagittarius but it doesn&rsquo;t matter":7,
                 "aries but it doesn&rsquo;t matter":9,
                 "capricorn but it doesn&rsquo;t matter":11,
                 "pisces but it doesn&rsquo;t matter":6,
                 "scorpio but it doesn&rsquo;t matter":1,
                 "leo":2,
                 "libra":3,
                 "cancer":5,
                 "virgo":8,
                 "scorpio":1,
                 "gemini":0,
                 "taurus":4,
                 "aries":9,
                 "pisces":6,
                 "aquarius":10,
                 "sagittarius":7,
                 "capricorn":11,
                 "scorpio and it matters a lot":1,
                 "leo and it matters a lot":2,
                 "aquarius and it matters a lot":10,
                 "cancer and it matters a lot":5,
                 "gemini and it matters a lot":0,
                 "pisces and it matters a lot":6,
                 "libra and it matters a lot":3,
                 "taurus and it matters a lot":4,
                 "sagittarius and it matters a lot":7,
                 "aries and it matters a lot":9,
                 "capricorn and it matters a lot":11,
                 "virgo and it matters a lot":8}
df["signs_code"] = df.sign.map(signs_mapping)
print(df.signs_code.value_counts())
print("*******************************************************")

#DRINKS
print(df.drinks.value_counts())
drinks_mapping = {"not at all": 0, "rarely": 1, "socially": 2, "often": 3, "very often": 4, "desperately": 5}
df["drinks_code"] = df.drinks.map(drinks_mapping)
print("*******************************************************")

#SMOKES
print(df.smokes.value_counts())
smokes_mapping = {"no": 0, "sometimes": 1, "when drinking": 2, "yes": 3, "trying to quit": 4}
df["smokes_code"] = df.smokes.map(smokes_mapping)
print("*******************************************************")

#DRUGS
print(df.drugs.value_counts())
drugs_mapping = {"never": 0, "sometimes": 1, "often": 2}
df["drugs_code"] = df.drugs.map(drugs_mapping)
print(df.drugs_code.value_counts())
print("*******************************************************")

#HEIGHT
print(df.height.value_counts())
height_mapping = {70.0:4, 68.0:3, 67.0:3, 72.0:4,
                  69.0:3, 71.0:4, 66.0:3, 64.0:3,
                  65.0:3, 73.0:4, 63.0:3, 74.0:4,
                  62.0:3, 75.0:4, 61.0:3, 60.0:3,
                  76.0:4, 77.0:4, 59.0:2, 78.0:4,
                  79.0:4, 58.0:2, 80.0:5, 95.0:6,
                  57.0:2, 83.0:5, 36.0:1, 81.0:5,
                  82.0:5, 84.0:5, 56.0:2, 55.0:2,
                  53.0:2, 54.0:2, 94.0:6, 91.0:6,
                  50.0:2, 43.0:1, 37.0:1, 48.0:1,
                  88.0:5, 8.0:0, 93.0:6, 4.0:0,
                  49.0:1, 1.0:0, 42.0:1, 86.0:5,
                  47.0:1, 87.0:5, 90.0:6, 52.0:2,
                  9.0:0, 51.0:2, 89.0:5, 6.0:0, 3.0:0,
                  92.0:6, 85.0:5, 26.0:0}
df["height_code"] = df.height.map(height_mapping)
print(df.height_code.value_counts())
print("*******************************************************")

#ETHNICITY
print(df.ethnicity.value_counts())
ethnicity_mapping = {"white":0, "asian":1, "hispanic / latin":2,
                     "black":3, "other":4, "middle eastern, black, native american, indian":3,
                     "asian, black, native american, indian, pacific islander, hispanic / latin": 1,
                     "middle eastern, indian, white":0, "asian, black, native american, indian":3,
                     "black, native american, indian, white":3}
df["ethnicity_code"] = df.ethnicity.map(ethnicity_mapping)
print(df.ethnicity_code.value_counts())
print("*******************************************************")

#JOB
print(df.job.value_counts())
print("*******************************************************")

#SEX
print(df.sex.value_counts())
sex_mapping = {"m": 0, "f": 1}
df["sex_code"] = df.sex.map(sex_mapping)
print(df.sex_code.value_counts())
print("*******************************************************")

#EDUCATION
print(df.education.value_counts())
education_mapping= {"graduated from college/university":3, "graduated from masters program":4,
                    "working on college/university":3, "working on masters program":4,
                    "graduated from two-year college":2, "graduated from high school":1,
                    "graduated from ph.d program":4,"graduated from law school":4,
                    "working on two-year college":2, "dropped out of college/university":1,
                    "working on ph.d program":4, "college/university":3, "graduated from space camp":3,
                    "dropped out of space camp":1, "graduated from med school":4,"working on space camp":3,
                    "working on law school":4, "two-year college":2, "working on med school":4,
                    "dropped out of two-year college":1, "dropped out of masters program":3,
                    "masters program":4, "dropped out of ph.d program":4, "dropped out of high school":0,
                    "high school":1, "working on high school":1, "space camp":3, "ph.d program":4,
                    "law school":4, "dropped out of law school":3, "dropped out of med school":3,
                    "med school":4}
df["education_code"]= df.education.map(education_mapping)
print("*******************************************************")

#INCOME
print(df.income.value_counts())
income_mapping= {-1: 0, 20000:1, 100000:8,80000:7, 30000:2,40000:3,
                 50000:4, 60000:5, 70000:6, 150000:9, 1000000:12,
                 250000:10, 500000:11}
df["income_code"] = df.income.map(income_mapping)
print("*******************************************************")

#BODY_TYPE
print(df.body_type.value_counts())
body_type_mapping= {"average":0, "fit":1, "athletic":2, "thin":3, "curvy":4,
                    "a little extra":5, "skinny":6, "full figured":7, "overweight":8,
                    "jacked":9, "used up":10, "rather not say":11}
df["body_type_code"] = df.body_type.map(body_type_mapping)
print("*******************************************************")

#DIET
print(df.diet.value_counts())
diet_mapping= {"mostly anything":0, "anything":0, "strictly anything":0,
               "mostly vegetarian":1, "mostly other":2, "strictly vegetarian":1,
               "vegetarian":1, "strictly other":2, "mostly vegan":3, "other":2,
               "strictly vegan":3, "vegan":3, "mostly kosher":4, "mostly halal":5,
               "strictly halal":5, "strictly kosher":4, "kosher":4, "halal":5}
df["diet_code"]= df.diet.map(diet_mapping)
print("*******************************************************")

#AGE
print(df.age.value_counts())
age_mapping= {26:1, 27:1, 28:1, 25:1, 29:1, 24:0, 30:2, 31:2,
              23:0, 32:2, 33:2, 22:0, 34:2, 35:3, 36:3, 37:3,
              38:3, 21:0, 39:3, 42:4, 40:5, 41:5, 20:0, 43:5,
              44:5, 45:6, 19:0, 46:6, 47:6, 48:6, 49:6, 50:7,
              51:7, 52:7, 18:0, 56:8, 54:7, 55:8, 57:8, 53:7,
              59:8, 58:8, 60:9, 61:9, 62:9, 63:9, 64:9, 65:10,
              66:10,67:10, 68:10, 69:10, 110:10, 109:10}
df["age_code"]= df.age.map(age_mapping)
print("*******************************************************")

#ESSAY
essay_cols = ["essay0","essay1","essay2","essay3","essay4","essay5","essay6","essay7","essay8","essay9"]
# Removing the NaNs
all_essays = df[essay_cols].replace(np.nan, '', regex=True)
# Combining the essays
df['all_essays'] = all_essays.apply(lambda x: ' '.join(x), axis=1)

df["essay_len"] = df.all_essays.apply(lambda x: len(x))
df["word_in_essay"] = df.all_essays.apply(lambda x: len(x.split(" ")))
df["avg_word_length"]=df['essay_len']/df['word_in_essay']
df["I_me_count"]= df.all_essays.apply(lambda x: (x.count("I") + x.count("me")))
print(df.columns)
print("*******************************************************")

df=df.dropna()
print(df.head())


#Classification is used to predict a discrete label. The outputs fall under a finite set of possible outcomes.
#Many situations have only two possible outcomes. This is called binary classification
#Multi-label classification is when there are multiple possible outcomes. It is useful for
#customer segmentation, image categorization, and sentiment analysis for understanding text.
#To perform these classifications, we use models like Naive Bayes, K-Nearest Neighbors, and SVMs.

#***********************************************************************************
#PREDICT ZODIAC SIGN BASED ON SMOKING, DRINKING, DRUGS, ESSAY_LEN,AVG_WORD_LEN
#KNN CLASSIFIER 1:
#***********************************************************************************
feature_data = df[['signs_code','smokes_code', 'drinks_code', 'drugs_code', 'essay_len', 'avg_word_length']]

X1 = feature_data.values
min_max_scaler = MinMaxScaler()
x_scaled1 = min_max_scaler.fit_transform(X1)

feature_data = pd.DataFrame(x_scaled1, columns=feature_data.columns)
X1 = feature_data[['smokes_code', 'drinks_code', 'drugs_code', 'essay_len', 'avg_word_length']].values
y1= np.ravel(df[["signs_code"]].values)

X_train1, X_test1, y_train1, y_test1 = train_test_split(X1,y1, test_size=0.2, random_state=1)
print(len(X1))
print(len(y1))

classifier_KNN = KNeighborsClassifier(n_neighbors=5)
classifier_KNN.fit(X_train1, y_train1)
y_pred1 = classifier_KNN.predict(X_test1)
print(confusion_matrix(y_test1, y_pred1))
print(classification_report(y_test1, y_pred1))

validation_accuracy=[]
error=[]
for i in range(1, 101):
    classifier_KNN= KNeighborsClassifier(n_neighbors=i)
    classifier_KNN.fit(X_train1, y_train1)
    guess = classifier_KNN.predict(X_test1)
    error.append(np.mean(guess != y_test1))
    validation_accuracy.append(classifier_KNN.score(X_test1,y_test1))
plt.figure(figsize=(12, 6))
#plt.plot(range(1, 101), error, color='red', linestyle='dashed', marker='o',
#        markerfacecolor='blue', markersize=10)
plt.plot(range(1, 101), validation_accuracy, color='green', linestyle='dashed', marker='o',
         markerfacecolor='purple', markersize=10)
plt.title('KNN Classifier Accuracy with K-value')
plt.xlabel('K Value')
plt.ylabel('Validation Accuracy')
plt.show()

#***********************************************************************************
#PREDICT SMOKING WITH EDUCATION LEVEL AND INCOME
# KNN CLASSIFIER2:
#***********************************************************************************
feature_data2 = df[['education_code', 'income_code', 'smokes_code']]
#print(feature_data.info)

X2 = feature_data2.values
min_max_scaler2 = MinMaxScaler()
x_scaled2 = min_max_scaler2.fit_transform(X2)

feature_data2 = pd.DataFrame(x_scaled2, columns=feature_data2.columns)
X2 = feature_data2[['education_code', 'income_code']].values
y2= np.ravel(df[["smokes_code"]].values)

X_train2, X_test2, y_train2, y_test2 = train_test_split(X2,y2, test_size=0.2, random_state=1)
print(len(X2))
print(len(y2))

classifier_KNN2 = KNeighborsClassifier(n_neighbors=28)
classifier_KNN2.fit(X_train2, y_train2)
y_pred2 = classifier_KNN2.predict(X_test2)
print(confusion_matrix(y_test2, y_pred2))
print(classification_report(y_test2, y_pred2))

validation_accuracy2=[]
error2=[]
for i in range(1, 101):
    classifier_KNN2= KNeighborsClassifier(n_neighbors=i)
    classifier_KNN2.fit(X_train2, y_train2)
    guess2 = classifier_KNN2.predict(X_test2)
    error2.append(np.mean(guess2 != y_test2))
    validation_accuracy2.append(classifier_KNN2.score(X_test2,y_test2))
plt.figure(figsize=(12, 6))
#plt.plot(range(1, 101), error2, color='red', linestyle='dashed', marker='o',
#        markerfacecolor='blue', markersize=10)
plt.plot(range(1, 101), validation_accuracy2, color='red', linestyle='dashed', marker='*',
         markerfacecolor='blue', markersize=10)
plt.title('KNN Classifier Accuracy with K-value')
plt.xlabel('K Value')
plt.ylabel('Validation Accuracy')
plt.show()

#***********************************************************************************
# PREDICT BODY_TYPE FROM DIET, INCOME & DRINKING
#KNN CLASSIFIER 3
#***********************************************************************************
feature_data3 = df[['diet_code', 'income_code', 'drinks_code', 'body_type_code']]
#print(feature_data.info)

X3 = feature_data3.values
min_max_scaler3 = MinMaxScaler()
x_scaled3 = min_max_scaler3.fit_transform(X3)

feature_data3 = pd.DataFrame(x_scaled3, columns=feature_data3.columns)
X3 = feature_data3[['diet_code', 'income_code', 'drinks_code']].values
y3= np.ravel(df[["body_type_code"]].values)

X_train3, X_test3, y_train3, y_test3 = train_test_split(X3,y3, test_size=0.2, random_state=1)
print(len(X3))
print(len(y3))
classifier_KNN3 = KNeighborsClassifier(n_neighbors=28)
classifier_KNN3.fit(X_train3, np.ravel(y_train3))
y_pred3 = classifier_KNN3.predict(X_test3)
print(confusion_matrix(y_test3, y_pred3))
print(classification_report(y_test3, y_pred3))
print(classifier_KNN3.score(X_test3, y_test3))


#***********************************************************************************
# PREDICT BODY TYPE FROM INCOME & EDUCATION
# SUPPORT VECTOR MACHINES CLASSIFIER
#***********************************************************************************
feature_data4 = df[['education_code', 'income_code', 'body_type_code']]
training_set_SVC,validation_set_SVC= \
                    train_test_split(feature_data4,test_size=0.2, random_state=1)

classifier_SVC=SVC(kernel='rbf', gamma= 100, C=100)
classifier_SVC.fit(training_set_SVC[['education_code', 'income_code']], \
                   training_set_SVC['body_type_code'])
print(classifier_SVC.score(validation_set_SVC[['education_code', 'income_code']],\
                           validation_set_SVC['body_type_code']))
y_pred4 = classifier_SVC.predict(validation_set_SVC[['education_code', 'income_code']])
print(confusion_matrix(validation_set_SVC['body_type_code'], y_pred4))
print(classification_report(validation_set_SVC['body_type_code'], y_pred4))

for gamma in range(1,10):
  for C in range(1,10):
    classifier_SVC=SVC(kernel='rbf', gamma= gamma, C=C)
    classifier_SVC.fit(training_set_SVC[['education_code', 'income_code']],\
                       training_set_SVC['body_type_code'])
    print("Gamma= " + str(gamma) + "and C= " + str(C) +" and Score= " + \
          str(classifier_SVC.score(validation_set_SVC[['education_code', 'income_code']], \
                                   validation_set_SVC['body_type_code'])))



#***********************************************************************************
# PREDICT EDUCATION FROM ESSAY_LEN
# SUPPORT VECTOR MACHINES CLASSIFIER2
#***********************************************************************************
feature_data5 = df[['essay_len']]
y5= np.ravel(df[['education_code']].values)
training_set_SVC2,validation_set_SVC2,training_label_SVC2,validation_label_SVC2=\
                    train_test_split(feature_data5,y5, test_size=0.2,random_state=1)
classifier_SVC2=SVC(kernel='rbf', gamma= 100, C=100)
classifier_SVC2.fit(training_set_SVC2.values.reshape(-1,1), training_label_SVC2)
y_pred5 = classifier_SVC2.predict(validation_set_SVC2)
print(confusion_matrix(validation_label_SVC2, y_pred5))
print(classification_report(validation_label_SVC2, y_pred5))

print(classifier_SVC2.score(validation_set_SVC2.values.reshape(-1,1), validation_label_SVC2))

for gamma in range(1,10):
  for C in range(1,10):
    classifier_SVC2=SVC(kernel='rbf', gamma= gamma, C=C)
    classifier_SVC2.fit(training_set_SVC2.values.reshape(-1,1), training_label_SVC2)    
    print("Gamma= " + str(gamma) + "and C= " + str(C) +" and Score= " + \
          str(classifier_SVC2.score(validation_set_SVC2.values.reshape(-1,1), \
                                    validation_label_SVC2)))




#***********************************************************************************
# PREDICT INCOME FROM ESSAY_LEN AND AGE
# K NEAREST NEIGHBOR REGRESSOR
#***********************************************************************************
feature_data6 = df[[ 'essay_len', 'age']]
y6=df[['income']]
training_set_KR,validation_set_KR, training_label_KR, validation_label_KR=\
                                   train_test_split(feature_data6,y6, test_size=0.2, random_state=100)

norm_data_KR= StandardScaler()
training_set_KR= norm_data_KR.fit_transform(training_set_KR)
validation_set_KR= norm_data_KR.transform(validation_set_KR)

# Create and train the model
model_KR=KNeighborsRegressor(n_neighbors=3, weights= 'distance')
model_KR.fit(training_set_KR, training_label_KR)

# Score the model on the train data
score_KR= model_KR.score(training_set_KR,training_label_KR)
#score(self, X, y, sample_weight=None)[source]
#Return the coefficient of determination R^2 of the prediction.
#The coefficient R^2 is defined as (1 - u/v), where u is the residual sum
#of squares ((y_true - y_pred) ** 2).sum() and v is the total sum of squares
#((y_true - y_true.mean()) ** 2).sum(). The best possible score is 1.0 and
#it can be negative (because the model can be arbitrarily worse).
#A constant model that always predicts the expected value of y, disregarding the
#input features, would get a R^2 score of 0.0.
print(score_KR)

# Score the model on the test data
score_test_KR= model_KR.score(validation_set_KR,validation_label_KR)
print(score_test_KR)
y_pred6 = model_KR.predict(validation_set_KR)
#print(confusion_matrix(validation_label_KR, y_pred6))
#print(classification_report(validation_label_KR, y_pred6))

#***********************************************************************************
# PREDICT GENDER FROM ESSAY CONTENT
# NAIVE BAYES CLASSIFIER 
#***********************************************************************************
feature_data7 = df[['all_essays', 'sex_code']]
#print(feature_data.info)

X7 = feature_data7['all_essays']
y7= np.ravel(df["sex_code"].values)

X_train7, X_test7, y_train7, y_test7 = train_test_split(X7,y7, test_size=0.2, random_state=1)
print(len(X7))
print(len(y7))

counter7= CountVectorizer()
counter7.fit(X_train7)
train_counts=counter7.transform(X_train7)
test_counts=counter7.transform(X_test7)

classifier_NB = MultinomialNB()
classifier_NB.fit(train_counts, y_train7)
y_pred7 = classifier_NB.predict(test_counts)
print(confusion_matrix(y_test7, y_pred7))
print(classification_report(y_test7, y_pred7))
print(classifier_NB.score(test_counts, y_test7))


#***********************************************************************************
# PREDICT INCOME FROM ESSAY_LEN AND AVG_WORD_LENGTH
#MULTIPLE LINEAR REGRESSION
#***********************************************************************************
feature_data8 = df[['income', 'essay_len', 'avg_word_length']]
#print(feature_data.info)

min_max_scaler8 = MinMaxScaler()
x_scaled8 = min_max_scaler8.fit_transform(feature_data8)

feature_data8 = pd.DataFrame(x_scaled8, columns=feature_data8.columns)
X8 = feature_data8[['essay_len', 'avg_word_length']]
y8= df[["income"]]

X_train8, X_test8, y_train8, y_test8 = train_test_split(X8,y8, test_size=0.2, random_state=1)
print(len(X8))
print(len(y8))
classifier_MLR = LinearRegression()
classifier_MLR.fit(X_train8, y_train8)
y_pred8 = classifier_MLR.predict(X_test8)

plt.scatter(y_test8,y_pred8, alpha=0.4)
plt.xlabel("The actual income")
plt.ylabel("The predicted income")
plt.title("Actual vs predicted income")
plt.show()

print(classifier_MLR.coef_)
print(classifier_MLR.score(X_train8, y_train8))
print(classifier_MLR.score(X_test8, y_test8))

#***********************************************************************************
# PREDICT AGE FROM I_OR_ME IN ESSAYS
# LINEAR REGRESSION
#***********************************************************************************
feature_data9 = df[['I_me_count', 'age']]
#print(feature_data.info)

min_max_scaler9 = MinMaxScaler()
x_scaled9 = min_max_scaler9.fit_transform(feature_data9)

feature_data9 = pd.DataFrame(x_scaled9, columns=feature_data9.columns)
X9 = feature_data9[['I_me_count']]
y9= df[["age"]]

X_train9, X_test9, y_train9, y_test9 = train_test_split(X9,y9, test_size=0.2, random_state=1)
print(len(X9))
print(len(y9))
classifier_LR = LinearRegression()
classifier_LR.fit(X_train9, y_train9)
y_pred9 = classifier_LR.predict(X_test9)

plt.scatter(y_test9,y_pred9, alpha=0.4)
plt.xlabel("The actual age")
plt.ylabel("The predicted age")
plt.title("Actual vs predicted age")
plt.show()

print(classifier_LR.coef_)
print(classifier_LR.score(X_train9, y_train9))
print(classifier_LR.score(X_test9, y_test9))
