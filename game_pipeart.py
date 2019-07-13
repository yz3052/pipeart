# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 16:02:16 2019

@author: tomyi
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
import time



# Data




i_data = pd.DataFrame([[1,1,1,3,1,5],
                       [2,1,3,4,3,5],
                       [3,1,5,2,3,5],
                       [4,3,3,5,5,5],
                       [5,4,1,5,3,5],
                       [6,2,5,4,5,5]],columns=['sequence','start_row','start_column','end_row','end_column','dim'])

i_data = pd.DataFrame([[1,1,1,4,2,7],
                       [2,1,5,2,6,7],
                       [3,1,6,3,5,7],
                       [4,2,2,4,4,7],
                       [5,2,3,4,3,7],
                       [6,4,1,6,5,7],
                       [7,5,4,7,7,7],
                       [8,5,6,7,2,7],
                       [9,7,1,6,2,7]
                       ],columns=['sequence','start_row','start_column','end_row','end_column','dim'])




i_data = pd.DataFrame([[1,1,1,8,1,9],
                       [2,1,2,6,7,9],
                       [3,1,3,4,3,9],
                       [4,2,5,6,2,9],
                       [5,2,8,5,7,9],
                       [6,4,7,9,7,9],
                       [7,6,8,9,3,9],
                       [8,7,2,8,4,9],
                       [9,8,3,9,1,9],
                       [10,8,8,9,6,9]
                       ],columns=['sequence','start_row','start_column','end_row','end_column','dim'])


i_data = pd.DataFrame([[1,1,9,6,1,10],
                       [2,1,10,2,7,10],
                       [3,2,6,3,10,10],
                       [4,3,3,6,8,10],
                       [5,4,3,10,3,10],
                       [6,5,9,8,6,10],
                       [7,6,5,7,1,10],
                       [8,6,9,9,2,10],
                       [9,7,5,10,10,10]],columns=['sequence','start_row','start_column','end_row','end_column','dim'])

o_data=np.zeros((i_data['dim'][0],i_data['dim'][0]))


# Matrix

for i in range(i_data.shape[0]):
    
    t_coord_x = i_data['start_row'][i] - 1
    t_coord_y = i_data['start_column'][i] - 1 
    
    o_data[t_coord_x,t_coord_y]=i_data['sequence'][i]
    
    t_coord_x2 = i_data['end_row'][i] - 1
    t_coord_y2 = i_data['end_column'][i] - 1 
    
    o_data[t_coord_x2,t_coord_y2]=i_data['sequence'][i]

# Visualization

t_color_str=['black','cyan','gold','pink','yellow', 'blue', 'red','white','orange','green','grey','purple']
t_cmap = ListedColormap(t_color_str[:i_data.shape[0]+1])

# Display matrix
plt.matshow(o_data,cmap=t_cmap)
plt.show()


###########
########### Algo
###########



# initiate occupied list
t_occupied=[[i_data['start_row'][i],i_data['start_column'][i]] for i in range(i_data.shape[0])] + \
            [[i_data['end_row'][i],i_data['end_column'][i]] for i in range(i_data.shape[0])]
t_occupied_start = [[i_data['start_row'][i],i_data['start_column'][i]] for i in range(i_data.shape[0])]
t_occupied_end = [[i_data['end_row'][i],i_data['end_column'][i]] for i in range(i_data.shape[0])]


# matrix cleanse
def route_process(route_ongoing,t_adj_all, t_occupied_start, t_occupied_end):
    
    x = route_ongoing
    
    # Step 1: find all possible next steps
    x = x.merge(t_adj_all,how='left',left_on=['curr_position_x','curr_position_y'],right_on=['1_x','1_y']).\
        drop(['1_x','1_y'],axis = 1)
        
    
    
    x['coord_str'] =  '['+x['2_x'].astype('str')+', '+x['2_y'].astype('str')+']'
    x['coord']  = x['coord_str'] .apply(lambda x: eval(x))
    
    
        ############################################3
    
    # Step 3: if we move into any prior movement (the not the immediately before), delete all relevant routes
    x = x.reset_index(drop=True)   
    
    x =  x.drop( x[x['route_list'].isin(x[[x['coord_str'][i] in str(x['route_list'][i])[:-12] for i in range(x.shape[0])]]['route_list'])].index , axis = 0) 
    
    
    # Step 2: do not move into starting points    
    x = x [~x['coord_str'].isin([str(i) for i in t_occupied_start])]
        
    

    
    # Step 3.5: do not move into occupied points due to prior movemwnt
    x = x.reset_index(drop=True)
    
    '''
    for i,j in x[['2_x','2_y']].drop_duplicates().iterrows():
        x= x.drop( x[(x['route_list'].str.contains('['+str(j['2_x'])+', '+str(j['2_y'])+']',regex=False))&\
                     (x['2_x']==j['2_x'])&(x['2_y']==j['2_y'])].index, axis = 0)
    '''    
        
    x =  x.drop( x[[x['coord_str'][i] in str(x['route_list'][i]) for i in range(x.shape[0])]].index , axis = 0) 
    x = x.reset_index(drop=True)
    print time.time()
    
    # Step 4: Tag "completed routes"
    x.loc[(x['2_x']==x['end_row'])&\
          (x['2_y']==x['end_column']),'completed_route']=True
    
    # Step 5: do not move into other's end points
    x = x.drop(    x[(x['completed_route']==False)&\
                     (x['coord'].apply(lambda x: x in t_occupied_end))].index, axis = 0 )
       
    
    # Step 6: update route_list and curr_position
    x['prev_position_x']=x['curr_position_x']
    x['prev_position_y']=x['curr_position_y']
    x['curr_position_x']=x['2_x']
    x['curr_position_y']=x['2_y']
    x['curr_position']=x['coord']
    x['route_list']=x['route_list'].astype(str).str[:-1]+','+x['curr_position'].astype(str)+']'
    
    
    # final step: 
    x = x.drop(['coord','2_x','2_y'],axis = 1)
    x = x.reset_index(drop=True)

    
    return x



# adjacent matrix
t_adj_mat1= pd.DataFrame([[i,j] for i in range(1,i_data['dim'][0]+1) for j in range(1,i_data['dim'][0]+1)],columns= ['1_x','1_y'])
t_adj_mat1['2_x'] = t_adj_mat1['1_x']-1
t_adj_mat1['2_y'] = t_adj_mat1['1_y']+0

t_adj_all=t_adj_mat1.append(\
                            pd.DataFrame({'1_x':t_adj_mat1['1_x'],'1_y':t_adj_mat1['1_y'],'2_x':t_adj_mat1['1_x']+0,'2_y':t_adj_mat1['1_y']-1})\
                            ).append(\
                            pd.DataFrame({'1_x':t_adj_mat1['1_x'],'1_y':t_adj_mat1['1_y'],'2_x':t_adj_mat1['1_x']+1,'2_y':t_adj_mat1['1_y']+0})\
                            ).append(\
                            pd.DataFrame({'1_x':t_adj_mat1['1_x'],'1_y':t_adj_mat1['1_y'],'2_x':t_adj_mat1['1_x']+0,'2_y':t_adj_mat1['1_y']+1}))

t_adj_all = t_adj_all[(t_adj_all>=1).all(axis=1)]
t_adj_all = t_adj_all[(t_adj_all<=i_data['dim'][0]).all(axis=1)]

del t_adj_mat1


# routes initiation
route=pd.DataFrame(zip(t_occupied_start, 
                       t_occupied_start,
                       [i[0] for i in t_occupied_start],
                       [i[1] for i in t_occupied_start],
                       [False]*len(t_occupied_start)),
                columns = ['route_list','curr_position','curr_position_x','curr_position_y',\
                           'completed_route'])
route.loc[:,'route_list']='['+route['route_list'].apply(lambda x: str(x))+']'
route=route.merge(i_data[['start_row','start_column','end_row','end_column']],how='left',left_on=['curr_position_x','curr_position_y'],right_on=['start_row','start_column'])



# develop routes

t_status = True 

while t_status:
    
    # get all next coordinates, without cleansing
    route_completed = route[route['completed_route']==True]
    route_ongoing = route[route['completed_route']==False]
    
    # pause loop if no more ongoing routes
    if route_ongoing.shape[0]==0:
        t_status=False
        continue
    
    # update route
    route_ongoing2 = route_process(route_ongoing, t_adj_all, t_occupied_start, t_occupied_end)
    route=route_ongoing2.append(route_completed)
    route=route.reset_index(drop=True)
    
    print route_completed.shape[0], route.shape[0]
    
    
#####
##### find the solution
#####

t_start_point = route.groupby(['start_column','start_row'],as_index=False)['start_column'].agg({'count':'count'}).sort_values('count',ascending=True).reset_index(drop=True)


for i in range(t_start_point.shape[0]): #iterate through different colors (starting points)
    
    if i==0:
        solution = route[(route['start_row']==t_start_point['start_row'][i])&
                     (route['start_column']==t_start_point['start_column'][i])][['route_list','start_row','start_column']]
        solution = solution.rename(columns = {'route_list':'route_list'+str(i),'start_row':'start_row'+str(i),'start_column':'start_column'+str(i)})
        solution.index = [0]*solution.shape[0]
        
        solution['agg_route']=solution.apply(lambda x: eval(x['route_list'+str(i)]) ,axis =1)
        #solution['agg_route_dict']=solution.apply(lambda x: {x[str(x['start_row'+str(i)])+"@"+str(x['start_column'+str(i)]):'route_list'+str(i)]} ,axis =1)
        
    else:
        t_route = route[(route['start_row']==t_start_point['start_row'][i])&
                     (route['start_column']==t_start_point['start_column'][i])][['route_list','start_row','start_column']]
        t_route = t_route.rename(columns = {'route_list':'route_list'+str(i),'start_row':'start_row'+str(i),'start_column':'start_column'+str(i)})
        t_route.index = [0]*t_route.shape[0]
        
        solution.index = [0]*solution.shape[0]
        solution = solution.join(t_route,how='outer')
        
        solution['agg_route']=solution.apply(lambda x: x['agg_route'] + eval(x['route_list'+str(i)]) ,axis =1)

        
        # delete all solutions that have routes overlapping with each other
        solution = solution.reset_index(drop = True)
        for j in range(solution.shape[0]):
            t_compare_1 = [str(a[0])+'@'+str(a[1]) for a in solution['agg_route'][j]]
            
            if len(set(t_compare_1)) != len(t_compare_1):
                solution = solution.drop(index = [j])
                
# visualize solution

t_solut_slice = solution.iloc[0,:]
o_data_solut=np.zeros((i_data['dim'][0],i_data['dim'][0]))


for i in range((len(t_solut_slice)-1)/3): #iterate through colors and construct data to be plotted    
    
    t_sequence_num = i_data[(i_data['start_row']==t_solut_slice['start_row'+str(i)])&\
                (i_data['start_column']==t_solut_slice['start_column'+str(i)])]['sequence']
    
    for j in eval(t_solut_slice['route_list'+str(i)]):
        o_data_solut[j[0]-1,j[1]-1] = t_sequence_num

t_color_str=['black','cyan','gold','pink','yellow', 'blue', 'red','white','orange','green','grey','purple']
t_cmap = ListedColormap(t_color_str[:i_data.shape[0]+1])

# Display matrix
plt.matshow(o_data_solut,cmap=t_cmap)
plt.show()
