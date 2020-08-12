import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


# load and investigate the data here:
df= pd.read_csv("tennis_stats.csv")
#df.info()
X= df[["ServiceGamesWon"]]
y=df[["Ranking"]]
plt.scatter(X,y)
lr=LinearRegression()
x_train,x_test,y_train,y_test=train_test_split(X,y,train_size=0.8, test_size=0.2)
lr.fit(x_train,y_train)
y_predict=lr.predict(x_test)
plt.plot(x_test, y_predict)
plt.show()
plt.figure()
plt.scatter(y_test,y_predict)
plt.show()


## perform single feature linear regressions here:
plt.figure()
X2= df[["BreakPointsOpportunities"]]
y2=df[["Winnings"]]
plt.scatter(X2,y2, c='red')
lr2=LinearRegression()
x_train2,x_test2,y_train2,y_test2=train_test_split(X2,y2,train_size=0.8, test_size=0.2)
lr2.fit(x_train2,y_train2)
y_predict2=lr2.predict(x_test2)
plt.plot(x_test2, y_predict2)
plt.show()
plt.figure()
plt.scatter(y_test2,y_predict2, c='red')
plt.show()

## perform two feature linear regressions here:
plt.figure()
X3= df[["ServiceGamesWon", "TotalPointsWon"]]
y3=df[["Winnings"]]
lr3=LinearRegression()
x_train3,x_test3,y_train3,y_test3=train_test_split(X3,y3,train_size=0.8, test_size=0.2)
lr3.fit(x_train3,y_train3)
y_predict3=lr3.predict(x_test3)
plt.scatter(y_test3,y_predict3, c='green')
plt.show()
