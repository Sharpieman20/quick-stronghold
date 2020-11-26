import numpy as np


chance_for_nine_or_more = 0

nine_or_more = 0
total = 0

for i in range(0, 100000):


    
    rods = (np.random.rand(16) > 0.5).sum()



    if rods > 8:
        nine_or_more += 1

    total += 1

print(nine_or_more / total)