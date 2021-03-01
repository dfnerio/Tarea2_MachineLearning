
import pandas as pd
import LR
import GR

# main

data = pd.read_csv('01-food-profit.csv')
x = data["x"]
y = data["y"]

print(LR.test(LR.train(x,y), 15))
