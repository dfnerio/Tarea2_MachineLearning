
# A01193624 Diego Frías Nerio
# A0

import pandas as pd
import LR
import GR

# main

data = pd.read_csv('dataset-1.csv')
test = pd.read_csv('test.csv')

data.head()
test.head()

features = ["x0", "mark1", "mark2"]
x = data[features]
y = data.accepted

testX = test[features]

print(LR.test(LR.train(x,y), testX, 0.7))