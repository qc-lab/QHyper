
import pandas as pd
import matplotlib.pyplot as plt

steps=50
for i in range(steps):
    
    res=pd.read_csv("probability_step"+str(i)+".csv").sort_values('result')
    
    res.plot(x='result', y='prop', kind='bar',ylim=(0, 0.05))
    
    plt.savefig("probability_stepr"+str(i)+".png")
    plt.close()
