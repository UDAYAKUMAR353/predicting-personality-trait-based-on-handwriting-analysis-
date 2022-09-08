import pandas as pd


train=pd.read_csv("personality.csv");

trainv=train.values

my_dict = {}

for i in range(len(trainv)):
    my_dict[trainv[i][0]]=trainv[0][1]



