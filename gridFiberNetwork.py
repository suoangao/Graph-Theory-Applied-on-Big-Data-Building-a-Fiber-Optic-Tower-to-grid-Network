import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree

# import two data set from csv
Phoenix_df = pd.read_csv('/Electricity Infrastructure/Phoenix_df.csv')
Nairobi_df = pd.read_csv('/Electricity Infrastructure/Nairobi_df.csv')

Phoenix_df = Phoenix_df[Phoenix_df.range > 16000]
Nairobi_df = Nairobi_df[Nairobi_df.range > 16000]

PN_cell_towers = pd.concat((Phoenix_df['lat'], Phoenix_df['lon']), axis=1)
NB_cell_towers = pd.concat((Nairobi_df['lat'], Nairobi_df['lon']), axis=1)

# ##### Phoenix Map to Grid Array ##### ############################################################################

# read in image data
grid1 = cv2.imread('Phoenix Power Grid.png')
print("Phoenix Grid")

# axis of grid color array
xindexArr1 = np.array([])
yindexArr1 = np.array([])

# iterate through pixels in 686x424 resolution image

for i in range(0, 424):
    for j in range(0, 686):

        # uncomment to turn pixels outside of color tolerance (+/- 25) white for extracted grid display
        if not (grid1.item(i,j,0) > 117 and grid1.item(i,j,0) < 167
                and grid1.item(i,j,1) > 70 and grid1.item(i,j,1) < 120 and grid1.item(i,j,2) > 86
                and grid1.item(i,j,2) < 136 and (grid1.item(i,j,0) is not 118) and (grid1.item(i,j,1) is not 118)
                and (grid1.item(i,j,2) is not 118) and (grid1.item(i,j,0) is not 119) and (grid1.item(i,j,1) is not 119)
                and (grid1.item(i,j,2) is not 119)):

            grid1[i, j] = [255, 255, 255]

        else:
            xindexArr1 = np.append(xindexArr1, j)
            yindexArr1 = np.append(yindexArr1, i)

# convert indexes into lat and long
xindexArr1 = xindexArr1 * 0.000320 + 33.509583
yindexArr1 = yindexArr1 * 0.000283 - 112.186336

Grid_array = np.column_stack((xindexArr1, yindexArr1))
# print(Grid_array)
print('Grid 1 Length', Grid_array.shape)

# this array contains the shortest distance from a tower to the power grid
shortDistArr1 = np.array([])

# shortest distance between the  cell tower(each one in the Phoenix cell array):
for index, row in PN_cell_towers.iterrows():
    sDist = np.amin(np.sqrt(((row['lat'] - Grid_array[:, 0]) * 110.90444)**2 + ((row['lon'] - Grid_array[:, 1]) * 93.45318)**2))
    sDistindex = np.argmin(np.sqrt(((row['lat'] - Grid_array[:, 0]) * 110.90444)**2 + ((row['lon'] - Grid_array[:, 1]) * 93.45318)**2))
    addRow = np.reshape(row, (1, 2))
    Grid_array = np.concatenate((Grid_array, addRow), axis=0)
    shortDistArr1 = np.append(shortDistArr1, sDist)

totalFoDistPN = np.sum(shortDistArr1)
print("toatlFoDistPN", totalFoDistPN)

# ##### Nairobi Map to Grid Array ##### #############################################################################
print('_'*77)

# read in image data
grid2 = cv2.imread('Nairobi Power Grid.png')
print("Nairobi Power Grid")

# axis of grid color array
xindexArr2 = np.array([])
yindexArr2 = np.array([])

# iterate through pixels in 660x422 resolution image
for q in range(0, 422):
    for r in range(0, 660):

        # turn pixels outside of grid white (zero tolerance) for extracted grid display
        if not ((grid2.item(q,r,0) is 84 and grid2.item(q,r,1) is 47 and grid2.item(q,r,2) is 0) or
                    (grid2.item(q,r,0) is 245 and grid2.item(q,r,1) is 166 and grid2.item(q,r,2) is 34)):
            grid2[q, r] = [255,255,255]
        else:
            xindexArr2 = np.append(xindexArr2, r)
            yindexArr2 = np.append(yindexArr2, q)

# now we convert the aixs to deg cordinate
xindexArr2 = xindexArr2 * 0.000805 - 1.162116
yindexArr2 = yindexArr2 * 0.000720 + 36.651600

Grid_array2 = np.column_stack((xindexArr2, yindexArr2))
# print(Grid_array2)
print('Grid 2 Length', Grid_array2.shape)

# this array contains the shortest distance from a tower in 33018 to the power grid
shortDistArr2 = np.array([])

# shortest distance between the  cell tower(each one in the Phoenix cell array):
for indexi, row in NB_cell_towers.iterrows():
    sDist2 = np.amin(np.sqrt(((row['lat'] - Grid_array2[:, 0]) * 110.57483725)**2 + ((row['lon'] - Grid_array2[:, 1]) * 111.29134198)**2))
    sDistindex2 = np.argmin(np.sqrt(((row['lat'] - Grid_array2[:, 0]) * 110.57483725)**2 + ((row['lon'] - Grid_array2[:, 1]) * 111.29134198)**2))
    addRow2 = np.reshape(row, (1, 2))
    Grid_array2 = np.concatenate((Grid_array2, addRow2), axis=0)
    shortDistArr2 = np.append(shortDistArr2, sDist2)

totalFoDistNB = np.sum(shortDistArr2)
print("toatlFoDistNB", totalFoDistNB)


# #################  Now it's time for some Graphics #####################################################
print('_'*77)
# #### plot of total distance of fiber optic
cityArr = ('Phoenix', 'Nairobi')
y_pos = np.arange(len(cityArr))
totalLen = [totalFoDistPN, totalFoDistNB]
colors = ['b', 'r']
plt.bar(y_pos, totalLen, align='center', alpha=0.5, color=colors)
plt.xticks(y_pos, cityArr)
plt.title('Total Fiber Optic Cable Length (>10mi range)')


# #### plot of power grid and cell tower locality
# grid = sns.JointGrid(xindexArr1, -1 * yindexArr1, space=0, size=6, ratio=50)
# grid.plot_joint(plt.scatter, color="g")

# #### plot of the extracted power grid

# Display extracted grid and save output to "PhoenixExtractedGrid.png"
# cv2.imshow('Phoenix Extracted Grid', grid1)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# cv2.imwrite('PhoenixExtractedGrid.png', grid1)

# display extracted grid and save output to "NairobiExtractedGrid.png"
# cv2.imshow('Nairobi Extracted Grid', grid2)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# cv2.imwrite('NairobiExtractedGrid.png', grid2)

# ####### This part we calculate what length of foc needed if we construct a cell tower network them selves

Pnmatrix1 = PN_cell_towers.as_matrix(columns=None)
i = 0
matrix = np.array([])

while i < Pnmatrix1.shape[0]: # 0-191
    move1 = Pnmatrix1[i, :]
    matrix = np.append(matrix, np.sqrt(((move1[0] - Pnmatrix1[:, 0]) * 110.90444)**2 + ((move1[1] - Pnmatrix1[:, 1]) * 93.45318)**2))
    i = i + 1

newmatrix = np.reshape(matrix, (Pnmatrix1.shape[0], Pnmatrix1.shape[0]))
sA = csr_matrix(newmatrix)
Tcsr = minimum_spanning_tree(sA)
Tcsr = Tcsr.toarray()
Tcsr = csr_matrix(Tcsr)
print(csr_matrix.sum(Tcsr))
cellnetPN = csr_matrix.sum(Tcsr)

Pnmatrix2 = NB_cell_towers.as_matrix(columns=None)
j = 0
matrix2 = np.array([])

while j < Pnmatrix2.shape[0]: # 0-191
    move2 = Pnmatrix2[j, :]
    matrix2 = np.append(matrix2, np.sqrt(((move2[0] - Pnmatrix2[:, 0]) * 110.57483725)**2 + ((move2[1] - Pnmatrix2[:, 1]) * 111.29134198)**2))
    j = j + 1

newmatrix2 = np.reshape(matrix2, (Pnmatrix2.shape[0], Pnmatrix2.shape[0]))

sA2 = csr_matrix(newmatrix2)
Tcsr2 = minimum_spanning_tree(sA2)
Tcsr2 = Tcsr2.toarray()
Tcsr2 = csr_matrix(Tcsr2)
print(csr_matrix.sum(Tcsr2))
cellnetNB = csr_matrix.sum(Tcsr2)

print('_'*77)
# #### plot of total distance of fiber optic vs self connection in phoenix
plt.figure()
cityArr = ('cell to power grid', 'cell network')
y_pos = np.arange(len(cityArr))
totalLen = [totalFoDistPN, cellnetPN]
colors2 = ['purple', 'orange']
plt.bar(y_pos, totalLen, align='center', alpha=0.5, color=colors2)
plt.xticks(y_pos, cityArr)
plt.title('Cell-To-Grid vs. Cell-To-Cell Cable Length (Phoenix)')

print('_'*77)
# #### plot of total distance of fiber optic vs self connection in phoenix
plt.figure()
cityArr = ('cell to power grid', 'cell network')
y_pos = np.arange(len(cityArr))
totalLen = [totalFoDistNB, cellnetNB]
colors3 = ['green', 'yellow']
plt.bar(y_pos, totalLen, align='center', alpha=0.5, color=colors3)
plt.xticks(y_pos, cityArr)
plt.title('Cell-To-Grid vs. Cell-To-Cell Cable Length (Nairobi)')

plt.show()
