import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plot as graph
from collections import defaultdict
from pygal_maps_world.maps import World

aut = pd.read_csv('authorInfo.csv')
art = pd.read_csv('articleInfo.csv')

tab_Mast = pd.merge(art, aut, on='Article No.', how='outer')
tab_Mast.fillna(0, inplace=True )
tab_Mast = tab_Mast.sort_values('Year')
tab_Mast.to_csv('tab_Mast.csv')

#graph.yearPub
#graph.yearCit()
#graph.countryPlot
graph.classification()


