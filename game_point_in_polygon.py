# -*- coding: utf-8 -*-
"""
Created on Sat Jul 13 16:55:48 2019

@author: tomyi
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
import time


# raw data

i_data = pd.DataFrame([[1,1,9,6,1,10],
                       [2,1,10,2,7,10],
                       [3,2,6,3,10,10],
                       [4,3,3,6,8,10],
                       [5,4,3,10,3,10],
                       [6,5,9,8,6,10],
                       [7,6,5,7,1,10],
                       [8,6,9,9,2,10],
                       [9,7,5,10,10,10]],columns=['sequence','start_row','start_column','end_row','end_column','dim'])

route = [[1,1],[1,2],[1,3],[2,3],[3,3],[3,4],[3,5],[3,6],[3,7],[3,8],[3,9],[4,9],[5,9],[6,9],[7,9],[8,9],[9,9],[9,8],
         [9,7],[9,6],[9,5],[9,4],[9,3],[8,3],[7,3],[7,4],[6,4],[5,4],[5,3],[5,2],[5,1]]
DIM = 10

# image data

o_data=np.zeros((10,10))

for i in range(len(route)):
    
    o_data[route[i][0]-1,route[i][1]-1]=1
    
# image

t_color_str=['black','cyan','gold','pink','yellow', 'blue', 'red','white','orange','green','grey','purple']
t_cmap = ListedColormap(t_color_str[:2])

# Display matrix
plt.matshow(o_data,cmap=t_cmap)
plt.show()


# algorithm


# create map
i_mapping = pd.DataFrame([ [i,j,i+1,j+1] for i in range(1,DIM+1) for j in range(1, DIM+1) ]+
                        [[i,j,i+0,j+1] for i in range(1,DIM+1) for j in range(1, DIM+1) ]+
                        [[i,j,i+1,j+0] for i in range(1,DIM+1) for j in range(1, DIM+1) ]+
                        [[i,j,i-1,j-1] for i in range(1,DIM+1) for j in range(1, DIM+1) ]+
                        [[i,j,i-1,j-0] for i in range(1,DIM+1) for j in range(1, DIM+1) ]+
                        [[i,j,i-0,j-1] for i in range(1,DIM+1) for j in range(1, DIM+1) ]+
                        [[i,j,i+1,j-1] for i in range(1,DIM+1) for j in range(1, DIM+1) ]+
                        [[i,j,i-1,j+1] for i in range(1,DIM+1) for j in range(1, DIM+1) ]+
                        [[i,j,i  ,j  ] for i in range(1,DIM+1) for j in range(1, DIM+1) ]
                          , columns = ['1_x','1_y','2_x','2_y'])
i_mapping = i_mapping[(i_mapping['2_x']>=1)&(i_mapping['2_x']<=10)]
i_mapping = i_mapping[(i_mapping['2_y']>=1)&(i_mapping['2_y']<=10)]


def verify_route(route, i_mapping, i_data):
    
    # if not enough edge points, we do not delete any rows
    t_route = [i[0] for i in route] + [i[1] for i in route]
    if t_route.count(1)+t_route.count(10)<=1:
        return True
            
    
    DIM = i_data['dim'][0]

    # find starting point
    route_set = {(i[0],i[1]) for i in route }
    t_fullset = {(i,j) for i in range(1,DIM+1) for j in range(1,DIM+1)}
    t_starting_point = list(t_fullset.difference(route_set))[0]
    
    # find all points
    t_status2 = True 
    t_count = 0
    t_1 = pd.DataFrame([list(t_starting_point)], columns = ['1_x','1_y'])
    
    while t_status2:
        
        # find next points
        t_1 = t_1.merge(i_mapping, how = 'left', on =['1_x','1_y'])[['2_x','2_y']].rename(columns = {'2_x':'1_x','2_y':'1_y'})
        # delete points on the route
        t_1['coord_str'] = t_1.apply(lambda x: '['+x['1_x'].astype(str)+', '+x['1_y'].astype(str)+']', axis = 1)
        t_1 = t_1[~t_1['coord_str'].apply(lambda x: x in str(route))]
        # update all the points found so far
        t_1 = t_1[['1_x','1_y']].drop_duplicates()
        
        if t_count == t_1.shape[0]:
            t_status2 = False
            break
        else:
            t_count = t_1.shape[0]
    
    
    appearance = list(i_data.merge(t_1, how = 'inner', left_on = ['start_row','start_column'], right_on = ['1_x','1_y'])['sequence'])+\
                list(i_data.merge(t_1, how = 'inner', left_on = ['end_row','end_column'], right_on = ['1_x','1_y'])['sequence'])
    
    for i in set(appearance):
        if appearance.count(i) == 1:
            return False
    
    return True


verify_route(route, i_mapping, i_data)


'''
o_data2=np.zeros((10,10))
t_1 = t_1.reset_index(drop=True)
for i in range(40):    
    o_data2[t_1['1_x'][i]-1,t_1['1_y'][i]-1]=1
    
# image

t_color_str=['black','cyan','gold','pink','yellow', 'blue', 'red','white','orange','green','grey','purple']
t_cmap = ListedColormap(t_color_str[:2])

# Display matrix
plt.matshow(o_data2,cmap=t_cmap)
plt.show()
'''