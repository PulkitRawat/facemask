from without_mask import wom
from with_mask import wm
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import numpy as np

x = np.r_[wm,wom]

labels = np.zeros(x.shape[0])
labels[2000:] = 1.0

output = {0:"mask on",1:"mask off"}

x_train, x_test, y_train, y_test = train_test_split(x,labels,test_size=0.25)
y_pred = SVC.predict(x_test)  
print(accuracy_score(y_test, y_pred))
