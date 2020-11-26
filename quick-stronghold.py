import numpy as np
import matplotlib.pyplot as plt
import matplotlib.markers as mark
from pathlib import Path
import scipy.stats as st

class Stronghold:

    def __init__(self, point, angle, dist, is_angle_from_origin):

        if angle is None:
            self.x = point.x
            self.y = point.y
            self.weight = 1
            return
        elif not is_angle_from_origin:
            self.angle = angle
            rad_angle = np.radians(90-angle)
            # print("cos " + str(np.cos(rad_angle)))
            self.x = point.x - dist * np.cos(rad_angle)
            self.y = point.y + dist * np.sin(rad_angle)
            self.angle_origin = np.degrees(self.calc_origin_angle_from_coords())
            # print(self)
        else:
            self.angle = angle
            rad_angle = np.radians(angle)
            self.x = -1 * dist * np.cos(rad_angle)
            self.y = dist * np.sin(rad_angle)

        angle = np.radians(90-angle)

        a = -np.tan(angle)

        b = -1

        c = a*(-point.x) + point.y

        numerator = abs(a * self.x + b * self.y + c)

        denom = np.sqrt(a**2 + b**2)

        dist = numerator / denom

        self.weight = 1 / (1 + (dist/THROW_1_DIST_WEIGHT)**2)

    def calc_origin_angle_from_coords(self):

        self.origin_dist = np.sqrt(self.x**2+self.y**2)

        prop = self.y / self.origin_dist

        return np.arcsin(prop)

        # self.x = x
        # self.y = y
        # self.dist = ...
        # self.angle = ...
        # self.angle_origin = 
        # self.dist_origin = ...
        # self.weight = ...

    def dist_from_origin(self):

        return np.sqrt(self.x**2 + self.y**2)

    def __str__(self):
        return str(Point(self.x, self.y))


class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def dist_from_origin(self):

        return np.sqrt(self.x**2 + self.y**2)
    
    def __str__(self):
        return "X:{} Y:{}".format(self.x, self.y)


MIN_DIST_FOR_EYE_TO_RISE = 12
MIN_DIST_FROM_ORIGIN = 16 * 32 * 1.25
MAX_DIST_FROM_ORIGIN = 16 * 32 * 2.25

NUM_TRIALS = 0
# NUM_TRIALS = 100000
# NUM_TRIALS = 1000000


BASE_WEIGHT = 0.0436332
DIST_ERROR_WEIGHTING = 32

def is_valid(point):
    if dist < MIN_DIST_FOR_EYE_TO_RISE:
        return False
    distFromSpawn = getDistFromSpawn(point)
    if distFromSpawn > MAX_DIST_FROM_ORIGIN:
        return False
    return True

def get_max_dist_for_stronghold_from_throw(deg_angle, point, d_o):
    # system of equations
    # spawn line and my line

    angle = np.radians(90-deg_angle)

    m_f = -np.tan(angle)
    # print(m_f)
    b_f = m_f*(-point.x) + point.y

    a = 1 + (m_f)**2
    b = 2 * m_f * b_f
    c = b_f**2 - d_o**2

    pos_x1 = (-b + np.sqrt(b**2 - 4*a*c))/(2*a)
    pos_x2 = (-b - np.sqrt(b**2 - 4*a*c))/(2*a)

    # print(pos_x1)

    # check which one is towards the stronghold and which is away
    # set the one towards it to furthest_x

    pos_y1 = m_f * (pos_x1 - point.x) + point.y
    pos_y2 = m_f * (pos_x2 - point.x) + point.y

    pos_point1 = Point(pos_x1, pos_y1)
    pos_point2 = Point(pos_x2, pos_y2)

    return (pos_point1,pos_point2)

def get_rel_space_size():
    # calc odds of this getting culled
    # how to calc?
    # % of scenarios in which stronghold 2 is valid * % of scenarios in which stronghold 3 is valid
    pass


def get_dist_between_points(point1, point2):

    return np.sqrt(((point1.x-point2.x)**2) + ((point1.y-point2.y)**2))

def gaussian(mu, stdev):
    return np.random.normal(mu, stdev)

def uniform(min, max):
    return np.random.uniform(min, max)






correct_point = None
if False:
    correct_x = 1085
    correct_z = -259

    correct_point = Point(correct_x, correct_z)
    

# input_x = -249.5
# input_z = 700
# input_f = 153

all_strongholds = []
x_vals = []
y_vals = []

axes = plt.axes()
axes.set_ylim(1500,-1500)
axes.set_xlim(-1500,1500)

total_d = 0
num_d = 0


closest_guess = None
top_guess = None
for i in range(0,NUM_TRIALS):

    strongholds = []

    base_angle = uniform(0,2*np.pi)

    for j in range(0, 3):

        x_val = np.cos(base_angle)
        y_val = np.sin(base_angle)
    
        vec_strength = uniform(MIN_DIST_FROM_ORIGIN, MAX_DIST_FROM_ORIGIN)

        x_val *= vec_strength
        y_val *= vec_strength

        x_vals.append(x_val)
        y_vals.append(y_val)

        stronghold_point = Point(x_val, y_val)

        stronghold = Stronghold(stronghold_point,None,None,None)

        strongholds.append(stronghold)

        base_angle += (2*np.pi/3)
        if base_angle > 2 * np.pi:
            base_angle -= 2 * np.pi
    
    stronghold = strongholds[0]

    d = strongholds[0].dist_from_origin()
    d_1 = strongholds[1].dist_from_origin()
    d_2 = strongholds[2].dist_from_origin()

    if d < d_1:
        if d < d_2:
            num_d += 1
            total_d += d

    all_strongholds.append(strongholds)

print(total_d / num_d)

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

throws = read_from_file()

print(throws)

ind = 0

colors = ["blue", "green", "blueviolet"]

total_inner_angle = 0

total_numerator = 0

total_hypot = 0

total_in_box = 0

for throw in throws:

    throw1_x = throw[0]
    throw1_z = throw[1]

    if len(throw) == 2:
        plt.scatter(throw1_x, throw1_z, marker=mark.MarkerStyle('D','full'), color='orange')
        continue

    throw1_f = throw[2]



    point = Point(throw1_x,throw1_z)

    angle = np.radians(90-throw1_f)

    a = -np.tan(angle)

    b = -1

    c = a*(-point.x) + point.y



    x_vals = []
    y_vals = []

    nearest_point_1 = None
    farthest_point_1 = None

    for x in range(-1500, 1500):
        y = (a*x + c) / (-b)

        temp_point = Point(x,y)

        origin_dist = temp_point.dist_from_origin()

        if origin_dist >= MIN_DIST_FROM_ORIGIN:
            if origin_dist <= MAX_DIST_FROM_ORIGIN:
                is_valid = False
                if throw1_f < 90 and throw1_f > 0:
                    if x < throw1_x:
                        is_valid = True
                elif throw1_f < -90 and throw1_f > -180:
                    if x > throw1_x:
                        is_valid = True
                elif throw1_f <= 0 and throw1_f > -90:
                    if (x > throw1_x):
                        is_valid = True
                else:
                    if (x < throw1_x):
                        is_valid = True
                if is_valid:
                    x_vals.append(temp_point.x)
                    y_vals.append(temp_point.y)
                    if farthest_point_1 is None:
                        farthest_point_1 = Point(x,y)
                    elif get_dist_between_points(farthest_point_1, point) < get_dist_between_points(temp_point, point):
                        farthest_point_1 = Point(x,y)
                    if get_dist_between_points(temp_point, point) > MIN_DIST_FOR_EYE_TO_RISE:
                        if nearest_point_1 is None:
                            nearest_point_1 = Point(x, y)
                        elif get_dist_between_points(nearest_point_1, point) > get_dist_between_points(temp_point, point):
                            nearest_point_1 = Point(x, y)
                
    print(nearest_point_1)
    print(farthest_point_1)

    plt.scatter(x_vals, y_vals, marker='o', color=colors[ind])

    plt.scatter(throw1_x, throw1_z, marker=mark.MarkerStyle('D','full'), color='black')

    if a == 0.0:
        a_T = None
    else:
        a_T = -1 / a

    b_T = -1

    c_nearest = a_T*(-nearest_point_1.x) + nearest_point_1.y

    c_farthest = a_T*(-farthest_point_1.x) + farthest_point_1.y

    # print(c_nearest)
    # print(c_farthest)

    for trial in range(0,NUM_TRIALS):

        strongholds = all_strongholds[trial]

        stronghold = strongholds[0]

        x = stronghold.x
        y = stronghold.y

        stronghold_point = Point(x,y)

        is_valid = False

        if get_dist_between_points(stronghold, point) < get_dist_between_points(strongholds[1], point):
            if get_dist_between_points(stronghold, point) < get_dist_between_points(strongholds[2], point):
                is_valid = True



        # if not is_valid:
        #     strongholds[0].weight = 0.0
        #     continue

        is_in_box = False

        if a_T is None:
            if nearest_point_1.x < farthest_point_1.x:
                x_lower_bound = nearest_point_1.x
                x_upper_bound = farthest_point_1.x
            else:
                x_lower_bound = farthest_point_1.x
                x_upper_bound = nearest_point_1.x
            
            if x >= x_lower_bound and x <= x_upper_bound:
                is_in_box = True
        else:
            nearest_perp_line_y_val_at_x = (a_T*x + c_nearest) / (-b_T)

            farthest_perp_line_y_val_at_x = (a_T*x + c_farthest) / (-b_T)


            if nearest_perp_line_y_val_at_x < farthest_perp_line_y_val_at_x:
                y_lower_bound = nearest_perp_line_y_val_at_x
                y_upper_bound = farthest_perp_line_y_val_at_x
            else:
                y_lower_bound = farthest_perp_line_y_val_at_x
                y_upper_bound = nearest_perp_line_y_val_at_x

            if y <= y_upper_bound and y >= y_lower_bound:
                is_in_box = True
            
        numerator = abs(a * stronghold_point.x + b * stronghold_point.y + c)

        total_numerator += numerator

        denom = np.sqrt(a**2 + b**2)

        dist = numerator / denom

        opposite = dist

        hypot = get_dist_between_points(point, stronghold_point)

        total_hypot += hypot
        
        if is_in_box:
            total_in_box += 1
            inner_angle = abs(np.arcsin(opposite / hypot))
            
        else:
            inner_angle = np.pi/2 + abs(np.arcsin(opposite / hypot))

        total_inner_angle += inner_angle
        
        # print(stronghold)
        # print(inner_angle)

        inner_zscore = inner_angle / BASE_WEIGHT



        # print(inner_zscore)

        inner_pct = st.norm.cdf(inner_zscore)

        if inner_pct < 0.5:
            print(inner_pct)

        # print(inner_pct)
        # print()

        strongholds[0].weight *= 1 - (2 * abs(0.5 - inner_pct))

        

        if ind == len(throws)-1 or len(throws[ind+1])<3:
            if top_guess is None:
                top_guess = strongholds[0]
            elif top_guess.weight < strongholds[0].weight:
                top_guess = strongholds[0]
            if correct_point is not None:
                if closest_guess is None:
                    closest_guess = stronghold
                elif get_dist_between_points(stronghold_point, correct_point) < get_dist_between_points(closest_guess, correct_point):
                    closest_guess = stronghold
    ind += 1








print(y_upper_bound)
print(y_lower_bound)









x_vals = []
y_vals = []

total_weight = 0
total_x = 0
raw_total_x = 0
total_y = 0
raw_total_y = 0
# print()

for trial in range(0,NUM_TRIALS):
    stronghold = all_strongholds[trial][0]

    total_weight += stronghold.weight

    raw_total_x += stronghold.x
    raw_total_y += stronghold.y

    total_x += stronghold.weight * stronghold.x
    total_y += stronghold.weight * stronghold.y

    # if (stronghold.weight > 0.3):
        # print(stronghold)
    x_vals.append(stronghold.x)
    y_vals.append(stronghold.y)
    



# plt.scatter(x_vals, y_vals, marker='o', color='yellow')


total_x = total_x / total_weight
total_y = total_y / total_weight

best_guess = Point(total_x,total_y)

x_vals = [best_guess.x]
y_vals = [best_guess.y]

plt.scatter(x_vals, y_vals, marker='x', color='red')

weight_in_range = 0

for trial in range(0,NUM_TRIALS):
    stronghold = all_strongholds[trial][0]

    if stronghold.x > best_guess.x - 5:
        if stronghold.x < best_guess.x + 5:
            if stronghold.y > best_guess.y - 5:
                if stronghold.y < best_guess.y + 5:
                    weight_in_range += stronghold.weight

confidence = weight_in_range / total_weight



print("\nGuess: ")
print("\t" + str(best_guess))
print("\tConfidence: {:.2%}".format(confidence))
print("\tTravel Distance: {}".format(get_dist_between_points(point, best_guess)))
print("\tTotal weight: {}".format(total_weight))
print("\tWeight in range: {}".format(weight_in_range))
if correct_point is not None:
    print("\tDist: " + str(get_dist_between_points(best_guess, correct_point)))


print("\nTop guess:")
print("\t" + str(top_guess))
if correct_point is not None:
    print("\tDist: " + str(get_dist_between_points(top_guess, correct_point)))
print("\tWeight: " + str(top_guess.weight))

if correct_point is not None:
    print("\nClosest guess:")
    print("\t" + str(closest_guess))
    print("\tDist: " + str(get_dist_between_points(closest_guess, correct_point)))
    print("\tWeight: " + str(closest_guess.weight))

print("\nAverage weight: " + str(total_weight / NUM_TRIALS))
print("\nAvg in box: " + str(total_in_box / NUM_TRIALS))
# print("\nAvg y: " + str(raw_total_y / NUM_TRIALS))






x_vals = []
y_vals = []

for trial in range(0,0):
    alt_stronghold_1 = all_strongholds[trial][1]

    x_vals.append(alt_stronghold_1.x)
    y_vals.append(alt_stronghold_1.y)


# plt.scatter(x_vals, y_vals, 'ob')




x_vals = []
y_vals = []

for trial in range(0,0):
    alt_stronghold_2 = all_strongholds[trial][2]

    x_vals.append(alt_stronghold_2.x)
    y_vals.append(alt_stronghold_2.y)


plt.scatter(x_vals, y_vals, marker='o', color='green')



x_vals = []
y_vals = []

for i in range(0,10000):
    min_x_val = MIN_DIST_FROM_ORIGIN * -1
    max_x_val = MIN_DIST_FROM_ORIGIN

    x_val = uniform(min_x_val, max_x_val)

    # MIN_DIST_FROM_ORIGIN^2 = x_val^2 + y_val^2
    # MIN_DIST_FROM_ORIGIN^2 - x_val^2 = y_val^2
    # sqrt(MIN_DIST_FROM_ORIGIN^2 - x_val^2) = y_val

    mult = round(uniform(0,1))

    if mult == 0:
        mult = -1

    y_val = mult * np.sqrt(MIN_DIST_FROM_ORIGIN**2 - x_val**2)

    x_vals.append(x_val)
    y_vals.append(y_val)

plt.scatter(x_vals, y_vals, marker='o', color='black')


x_vals = []
y_vals = []

for i in range(0,10000):
    min_x_val = MAX_DIST_FROM_ORIGIN * -1
    max_x_val = MAX_DIST_FROM_ORIGIN

    x_val = uniform(min_x_val, max_x_val)

    # MIN_DIST_FROM_ORIGIN^2 = x_val^2 + y_val^2
    # MIN_DIST_FROM_ORIGIN^2 - x_val^2 = y_val^2
    # sqrt(MIN_DIST_FROM_ORIGIN^2 - x_val^2) = y_val

    mult = round(uniform(0,1))

    if mult == 0:
        mult = -1

    y_val = mult * np.sqrt(MAX_DIST_FROM_ORIGIN**2 - x_val**2)

    x_vals.append(x_val)
    y_vals.append(y_val)

plt.scatter(x_vals, y_vals, marker='o', color='black')



plt.savefig("strongholds.png",dpi=2**9)
