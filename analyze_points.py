import pandas as pd
from pathlib import Path
import math

IDEAL_DIST = 216

RATIO_TO_USE = 1.0
VEHICLE_RATIO = 1.61
BASE_SPEED = 7.14
VEHICLE_SPEED = 11.5

PORTAL_X = 10
PORTAL_Z = 8

END_X = 120
END_Z = 90

dist_back = int(math.sqrt((END_X-PORTAL_X)**2 + (END_Z-PORTAL_Z)**2))
dist_to_blind = IDEAL_DIST-int(math.sqrt(END_X**2+END_Z**2))

throw_dist_df = pd.read_json(Path.cwd() / 'points.json')

for x in ['25','50','100']:
    col_name = 'wgt_dist_{}'.format(x)

    throw_dist_df[col_name] -= throw_dist_df['x'] * 3
    # throw_dist_df[col_name] *= VEHICLE_RATIO
    # throw_dist_df[col_name] += throw_dist_df['x'] * RATIO_TO_USE
# throw_dist_df[]

col = 'wgt_dist_50'
avg_col = 'avg_dist_50'

throw_dist_df.sort_values(by=col,ascending=True,inplace=True)

throw_dist_df = throw_dist_df[['x',col,avg_col]]

# print(throw_dist_df.index)

# throw_dist_df.at[220,col] += throw_dist_df.at[220,'x'] * RATIO_TO_USE
# throw_dist_df.at[0,col] /= VEHICLE_RATIO
throw_dist_df['seconds'] = 0 # throw_dist_df['avg_dist_25'] / BASE_SPEED
dist = int(math.sqrt(PORTAL_X**2 + PORTAL_Z**2))


my_dist_to_travel = throw_dist_df.at[dist,avg_col] + dist_back

throw_dist_df.at[dist,'seconds'] = my_dist_to_travel / BASE_SPEED
throw_dist_df.at[IDEAL_DIST,'seconds'] = (dist_to_blind + throw_dist_df.at[IDEAL_DIST,avg_col]) / BASE_SPEED




print(throw_dist_df.loc[throw_dist_df['x'] == dist])
print(throw_dist_df.loc[throw_dist_df['x'] == IDEAL_DIST])
