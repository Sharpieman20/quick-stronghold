from random import random

NUM_RUNS = 1000

NUM_PER = 1000

count = 0

for i in range(0,NUM_RUNS * NUM_PER):
    num_filled = 0
    for j in range(0,10):
        if random() < 0.10:
            num_filled += 1
    if num_filled == 0:
        count += 1

print(float(count) / (NUM_RUNS * NUM_PER))