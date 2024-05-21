"""
File: main.py
Author: Kishor
Date: 7th June 2023
Description: K Nearest Neighboor Machine Learning Model To Predict The Weather With 75.7% Accuracy.

"""

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from collections import Counter
 
df = pd.read_csv('./Data/seattle-weather.csv')
 
X = df.iloc[:, :-1]
y = df.iloc[:, -1]
 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, random_state=0)



class KNN:
    
    
    def __init__(self,k,preci,mfloatem,maxtem,wind):
    	
    	try:
    		self.k=k
	    	self.pr=float(preci.iloc[0])
	    	self.it=float(mfloatem.iloc[0])
	    	self.at=float(maxtem.iloc[0])
	    	self.w=float(wind.iloc[0])
    	except:
	    	self.k=k
	    	self.pr=preci
	    	self.it=mfloatem
	    	self.at=maxtem
	    	self.w=wind
    	
    def edis(self,xpr,xit,xat,xw):
    	distance=(((self.pr-xpr)**2)+((self.it-xit)**2)+((self.at-xat)**2)+((self.w-xw)**2))**(0.5)
    	return distance
    	
    def fit(self):
    	D=[]
    	for i in range(len(X_train)):
    		xpr=float(X_train[i:i+1].precipitation.iloc[0])
    		xit=float(X_train[i:i+1].temp_min.iloc[0])
    		xat=float(X_train[i:i+1].temp_max.iloc[0])
    		xw=float(X_train[i:i+1].wind.iloc[0])
    		d=self.edis(xpr,xit,xat,xw)
    		y=y_train[i:i+1].iloc[0]
    		D.append(d)
    	return D
    	
    def predict(self):
    	D=self.fit()
    	minV=sorted(D)[:self.k]
    	minY=[]
    	for i in minV:
    		id=D.index(i)
    		minY.append(y_train[id:id+1].iloc[0])
    	
    	
    	
    	result=Counter(minY).most_common(1)[0][0]
    	
    	return result
    	
 
def Test():
	k=3
	X=X_test
	y=y_test
	m=0
	n=0
	for i in range(len(X)):
		preci=X[i:i+1].precipitation
		mfloatem=X[i:i+1].temp_min
		maxtem=X[i:i+1].temp_max
		wind=X[i:i+1].wind
		
		expect=y[i:i+1].iloc[0]
		
		z=KNN(k,preci,mfloatem,maxtem,wind)
		predict=z.predict()
		n+=1
		if(expect==predict):
			m+=1
		print("Expect : ",expect,"- - Predict : ",predict)
	Accuracy=round((m/n)*100,1)
	print("Accuracy : ",Accuracy,"%")


def User():
	print("Hi ! I Am Weather Predictor...")
	xit=float(input("Minimum Temperature : "))
	xat=float(input("Maximum Temperature : "))
	xw=float(input("How About Wind : "))
	xpr=float(input("Precipitation : "))
	k=3
	z=KNN(k,xpr,xit,xat,xw)
	print("Today Weather May Be ",z.predict(),"...")
		
User()
 
