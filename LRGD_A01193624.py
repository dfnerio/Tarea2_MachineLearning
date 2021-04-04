
# A01193624 Diego Frías Nerio
# A0

import pandas as pd
import LR
import GR

# main

data = pd.read_csv('dataset-1.csv')
data.head()
features = ["x0", "mark1", "mark2"]
x = data[features]
y = data.accepted

print(LR.test(LR.train(x,y), x, 0.7))
