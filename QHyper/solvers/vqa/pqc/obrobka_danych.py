
import pandas as pd
import matplotlib.pyplot as plt

steps=50
for i in range(steps):
    
    res=pd.read_csv("probability_step"+str(i)+".csv")
    
    res.plot(x='energy', y='prop', kind='bar',ylim=(0, 0.05))
    
    plt.savefig("probability_step"+str(i)+".png")
    plt.close()
