
import pandas as pd
import matplotlib.pyplot as plt
res=pd.read_csv(".\demo\wyniki\probability_step0.csv").sort_values('energy')
print(res.energy)
steps=0
for i in range(steps):
    
    res=pd.read_csv("probability_step"+str(i)+".csv").sort_values('result')
    print(res.results)
    res.plot(x='result', y='prop', kind='bar',ylim=(0, 0.3))
    
    plt.savefig("probability_stepr"+str(i)+".png")
    plt.close()
