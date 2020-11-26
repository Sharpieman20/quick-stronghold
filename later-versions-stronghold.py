import numpy as np
import matplotlib.pyplot as plt
import matplotlib.markers as mark
from pathlib import Path
import scipy.stats as st
import pandas as pd
import sys

class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def dist_from_origin(self):

        return np.sqrt(self.x**2 + self.y**2)
    
    def __str__(self):
        return "X:{} Y:{}".format(self.x, self.y)

MIN_DIST_FOR_EYE_TO_RISE = 12
MIN_DIST_FROM_ORIGIN = 16 * 88
MAX_DIST_FROM_ORIGIN = 16 * 168

num_spawns = 0

# for i in range(0, 88):
#     for j in range(0, 168):
#         p = Point(i*16, j*16)
#         dist = p.dist_from_origin()
#         if dist > MIN_DIST_FROM_ORIGIN:
#             if dist < MAX_DIST_FROM_ORIGIN:
#                 num_spawns += 4

# print(num_spawns)

# NUM_TRIALS = 1000
NUM_TRIALS = int(sys.argv[1])

BASE_WEIGHT = 0.01
BASE_WEIGHT /= 16
BASE_WEIGHT /= 4
DIST_ERROR_WEIGHTING = 32

def read_from_file():

    fil = Path.cwd() / "input.txt"

    txt = fil.read_text()

    txt = txt.rstrip()

    ind = 0

    ret = []

    for line in txt.split("\n"):
        for item in line.split(" "):
            if ind % 3 == 0:
                x = item
                ind = 0
            elif ind == 1:
                y = item
            elif ind == 2:
                f = item
                ret.append((float(x),float(y),float(f)))
            ind += 1
            # print("|" + item + "|")
    if not ind % 3 == 0:
        ret.append((float(x),float(y)))

    return ret

def get_dist_between_points(point1, point2):

    return np.sqrt(((point1.x-point2.x)**2) + ((point1.y-point2.y)**2))

def gaussian(mu, stdev):
    return np.random.normal(mu, stdev)


all_strongholds = []
x_vals = []
y_vals = []

axes = plt.axes()
axes.set_ylim(MAX_DIST_FROM_ORIGIN,-MAX_DIST_FROM_ORIGIN)
axes.set_xlim(-MAX_DIST_FROM_ORIGIN,MAX_DIST_FROM_ORIGIN)


# MAKING BLACK CIRCLES
for i in range(0,10000):
    for dist in (MIN_DIST_FROM_ORIGIN, MAX_DIST_FROM_ORIGIN):
        min_x_val = dist * -1
        max_x_val = dist

        x_val = np.random.uniform(min_x_val, max_x_val)

        mult = round(np.random.uniform(0,1))

        if mult == 0:
            mult = -1

        y_val = mult * np.sqrt(dist**2 - x_val**2)

        x_vals.append(x_val)
        y_vals.append(y_val)

plt.scatter(x_vals, y_vals, marker='o', color='black')

# TODO: Vectorize stronghold gen code


df = pd.DataFrame()

df['angle'] = np.random.uniform(low=0, high=2*np.pi, size=NUM_TRIALS)

df['weight'] = 1

for j in range(0, 3):

    df['angle_{}'.format(j)] = df['angle'] + j * 2 * np.pi / 3

    df['strength_{}'.format(j)] = np.random.uniform(MIN_DIST_FROM_ORIGIN, MAX_DIST_FROM_ORIGIN, size=NUM_TRIALS)

    x_name = 'x_{}'.format(j)

    df[x_name] = np.cos(df['angle_{}'.format(j)])

    df[x_name.format(j)] *= df['strength_{}'.format(j)]

    df['{}_mod'.format(x_name)] = df[x_name] % 16

    df[x_name] -= df['{}_mod'.format(x_name)]

    df[x_name] += 8

    y_name = 'y_{}'.format(j)

    df[y_name] = np.sin(df['angle_{}'.format(j)])

    df[y_name] *= df['strength_{}'.format(j)]

    df['{}_mod'.format(y_name)] = df[y_name] % 16

    df[y_name] -= df['{}_mod'.format(y_name)]

    df[y_name] += 8


throws = read_from_file()

ind = 0

colors = ["blue", "green", "blueviolet"]

# TODO: Vectorize throw calculations (iterating per possible stronghold)
for throw in throws:

    throw_x = throw[0]
    throw_z = throw[1]

    if len(throw) < 3:
        plt.scatter(throw_x, throw_z, marker=mark.MarkerStyle('D','full'), color='orange')
        continue

    throw_f = throw[2]



    point = Point(throw_x,throw_z)

    angle = np.radians(90-throw_f)

    a = -np.tan(angle)

    b = -1

    c = a*(-point.x) + point.y



    x_vals = []
    y_vals = []

    nearest_point = None
    farthest_point = None

    for x_sub in range(-MAX_DIST_FROM_ORIGIN*10, MAX_DIST_FROM_ORIGIN*10):
        x = x_sub/10
        y = (a*x + c) / (-b)

        temp_point = Point(x,y)

        origin_dist = temp_point.dist_from_origin()

        if origin_dist >= MIN_DIST_FROM_ORIGIN:
            if origin_dist <= MAX_DIST_FROM_ORIGIN:
                is_valid = False
                if throw_f <= 90 and throw_f > 0:
                    if x < throw_x:
                        is_valid = True
                elif throw_f <= -90 and throw_f > -180:
                    if x > throw_x:
                        is_valid = True
                elif throw_f <= 0 and throw_f > -90:
                    if (x > throw_x):
                        is_valid = True
                else:
                    if (x < throw_x):
                        is_valid = True
                if is_valid:
                    x_vals.append(temp_point.x)
                    y_vals.append(temp_point.y)
                    if farthest_point is None:
                        farthest_point = Point(x,y)
                    elif get_dist_between_points(farthest_point, point) < get_dist_between_points(temp_point, point):
                        farthest_point = Point(x,y)
                    if get_dist_between_points(temp_point, point) > MIN_DIST_FOR_EYE_TO_RISE:
                        if nearest_point is None:
                            nearest_point = Point(x, y)
                        elif get_dist_between_points(nearest_point, point) > get_dist_between_points(temp_point, point):
                            nearest_point = Point(x, y)
                

    plt.scatter(x_vals, y_vals, marker='o', color=colors[ind])

    plt.scatter(throw_x, throw_z, marker=mark.MarkerStyle('D','full'), color='orange')

    if a == 0.0:
        a_T = None
    else:
        a_T = -1 / a

    b_T = -1

    c_nearest = a_T*(-nearest_point.x) + nearest_point.y

    c_farthest = a_T*(-farthest_point.x) + farthest_point.y

    if nearest_point.x < farthest_point.x:
        x_lower_bound = nearest_point.x
        x_upper_bound = farthest_point.x
    else:
        x_lower_bound = farthest_point.x
        x_upper_bound = nearest_point.x



    df['nearest_perp_line_y_val_at_x'] = (a_T*df['x_0'] + c_nearest) / (-b_T)

    df['farthest_perp_line_y_val_at_x'] = (a_T*df['x_0'] + c_farthest) / (-b_T)

    df['is_nearest_larger'] = df['nearest_perp_line_y_val_at_x'] > df['farthest_perp_line_y_val_at_x']

    df['y_upper_bound'] = df['is_nearest_larger'] * df['nearest_perp_line_y_val_at_x']

    df['y_upper_bound'] += (1 - df['is_nearest_larger']) * df['farthest_perp_line_y_val_at_x']

    df['y_lower_bound'] = (1-df['is_nearest_larger']) * df['nearest_perp_line_y_val_at_x']

    df['y_lower_bound'] += df['is_nearest_larger'] * df['farthest_perp_line_y_val_at_x']

    


    for j in range(0,3):
        df['dist_{}'.format(j)] = np.sqrt((df['x_{}'.format(j)]-throw_x)**2 + (df['y_{}'.format(j)]-throw_z)**2)
    
    df['is_valid'] = (df['dist_0'] < df['dist_1']) & (df['dist_0'] < df['dist_2'])

    df['weight'] *= df['is_valid']

    if a_T is None:
        df['in_box'] = (df['x_0'] >= x_lower_bound) & (df['x_0'] <= x_upper_bound)
    else:
        df['in_box'] = (df['y_0'] >= df['y_lower_bound']) & (df['y_0'] <= df['y_upper_bound'])

    df['inner_angle'] = (1 - df['in_box']) * np.pi/2
    
    df['numerator'] = np.abs(a * df['x_0'] + b * df['y_0'] + c)

    df['denom'] = np.sqrt(a**2 + b**2)

    df['opposite'] = df['numerator'] / df['denom']

    df['hypot'] = df['dist_0']

    df['ratio'] = df['opposite'] / df['hypot']

    df['inner_angle'] += np.abs(np.arcsin(df['ratio']))

    df['inner_zscore'] = df['inner_angle'] / BASE_WEIGHT

    print(df['inner_zscore'].max())

    df['inner_pct'] = st.norm.cdf(df['inner_zscore'])

    print(st.norm.cdf(7.2))

    print(df['inner_pct'].min())


    df['weight'] *= 1 - (2 * np.abs(0.5 - df['inner_pct']))

    ind += 1

    print(len(df))
    print(df['weight'].min())

    df.replace(to_replace={'weight':0.0},value=np.nan,inplace=True)

    print(df['weight'].isna().mean())

    df.dropna(subset=['weight'],inplace=True)

    print(len(df))
    

df['weighted_x'] = df['weight'] * df['x_0']

df['weighted_y'] = df['weight'] * df['y_0']

avg_x = df['weighted_x'].sum()
avg_y = df['weighted_y'].sum()
total_weight = df['weight'].sum()

avg_x /= total_weight
avg_y /= total_weight

best_guess = Point(avg_x,avg_y)

plt.scatter(df['x_0'], df['y_0'], marker='x', color='yellow')

plt.scatter(best_guess.x, best_guess.y, marker='x', color='red')

df['in_range'] = (df['x_0'] > best_guess.x - 5) & (df['x_0'] < best_guess.x + 5) & (df['y_0'] > best_guess.y - 5) & (df['y_0'] < best_guess.y + 5)

df['weight_in_range'] = df['in_range'] * df['weight']

weight_in_range = df['weight_in_range'].sum()

confidence = weight_in_range / total_weight



print("Guess: ")
print("\t" + str(best_guess))
print("\tConfidence: {:.2%}".format(confidence))
print("\tWeight in range: {}".format(weight_in_range))
print("\tTotal weight: {}".format(total_weight))

avg_weight = df['weight'].sum() / NUM_TRIALS


res = df.groupby(by=['x_0','y_0']).sum()

res = res.sort_values('weight',ascending=False)

res.reset_index()

pd.options.display.max_rows = 100

# res['rank'] = 

# print(res.loc[1128.0]['weight'])

print(res['weight'].head(25))

plt.savefig("strongholds.png",dpi=2**9)
