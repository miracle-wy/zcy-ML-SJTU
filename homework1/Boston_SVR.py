import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error

#数据导入
data=pd.read_csv('.\housing.csv')
X=pd.read_csv('.\housing.csv',usecols=[*range(0,12)])
y=pd.read_csv('.\housing.csv',usecols=[*range(12,13)])

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

#数据标准化处理
ss_X = StandardScaler()
ss_y = StandardScaler()
X_train = ss_X.fit_transform(X_train)
X_test = ss_X.transform(X_test)
y_train = ss_y.fit_transform(y_train.values.reshape(-1,1))
y_test = ss_y.transform(y_test.values.reshape(-1,1))

#回归训练与预测

#1.线性核函数
linear_svr = SVR(kernel='linear')
linear_svr.fit(X_train,y_train)
linear_svr_y_predict = linear_svr.predict(X_test)
#2.多项式核函数
poly_svr = SVR(kernel='poly')
poly_svr.fit(X_train,y_train)
poly_svr_y_predict = poly_svr.predict(X_test)
#3.高斯核函数
rbf_svr = SVR(kernel='rbf')
rbf_svr.fit(X_train,y_train)
rbf_svr_y_predict = rbf_svr.predict(X_test)

#性能评估
#1.线性核函数
print('R-squared value of linear SVR is',linear_svr.score(X_test,y_test))
print('the MSE of linear SVR is',mean_squared_error(ss_y.inverse_transform(y_test),ss_y.inverse_transform(linear_svr_y_predict)))
print('the RMSE of linear SVR is',np.sqrt(mean_squared_error(ss_y.inverse_transform(y_test),ss_y.inverse_transform(linear_svr_y_predict))))
print('the MAE of linear SVR is',mean_absolute_error(ss_y.inverse_transform(y_test),ss_y.inverse_transform(linear_svr_y_predict)))

#2.多项式核函数
print('R-squared value of Poly SVR is',poly_svr.score(X_test,y_test))
print('the MSE of Poly SVR is',mean_squared_error(ss_y.inverse_transform(y_test),ss_y.inverse_transform(poly_svr_y_predict)))
print('the RMSE of Poly SVR is',np.sqrt(mean_squared_error(ss_y.inverse_transform(y_test),ss_y.inverse_transform(poly_svr_y_predict))))
print('the MAE of Poly SVR is',mean_absolute_error(ss_y.inverse_transform(y_test),ss_y.inverse_transform(poly_svr_y_predict)))

#3.高斯核函数
print ('R-squared value of RBF SVR is',rbf_svr.score(X_test,y_test))
print ('the MSE of RBF SVR is',mean_squared_error(ss_y.inverse_transform(y_test),ss_y.inverse_transform(rbf_svr_y_predict)))
print ('the RMSE of RBF SVR is',np.sqrt(mean_squared_error(ss_y.inverse_transform(y_test),ss_y.inverse_transform(rbf_svr_y_predict))))
print ('the MAE of RBF SVR is',mean_absolute_error(ss_y.inverse_transform(y_test),ss_y.inverse_transform(rbf_svr_y_predict)))
