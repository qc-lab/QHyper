
import pandas as pd
import matplotlib.pyplot as plt
#res=pd.read_csv(".\demo\wynikirob\probability_step0.csv").sort_values('energy')
#print(res.energy)
steps=30
for i in range(steps):
    
    res=pd.read_csv(".\demo\wynikirob\probability_step"+str(i+1)+".csv")
    #.sort_values('result')

    res.plot(x='result', y='prop', kind='bar',ylim=(0, 0.3))
    
    plt.savefig(".\demo\wynikirob\probability_stepr"+str(i+1)+".png")
    plt.close()
