from random import random

NUM_RUNS = 1000

NUM_PER = 1000

count = 0

for i in range(0,NUM_RUNS * NUM_PER):
    for j in range(0,3):
        if random() < 0.2:
            count += 1
            break

print(float(count) / (NUM_RUNS * NUM_PER))