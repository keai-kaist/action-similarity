import numpy as np
import pandas
import matplotlib.pyplot as plt
from matplotlib import pyplot

"C:\experiments"
df = pandas.read_excel("C:\\Users\\jongwhoa\\OneDrive - kaist.ac.kr\\수업\\CS492 그래프 머신러닝 마이닝\텀프\\3. project final\\experiment1.xlsx")
data = df.to_numpy()
vals = data[:].T
#means = data[-1]


print(vals)
fig, ax = plt.subplots()
ax.imshow(vals, aspect='auto', cmap="Blues")
#ax.colorbar()
for i in range(len(vals)):
    for j in range(len(vals[i])):
        ax.text(j,i, np.round(vals[i, j], decimals=3),ha='center',va='center')
y_ticklabels = ["hand signal","jump","squat","walk","PT","stop"]
x_ticklabels = ["hand signal","jump","squat","walk","PT","stop", "mean wo gt"]

ax.set_xticks(np.arange(len(x_ticklabels)))
ax.set_xticklabels(x_ticklabels)
ax.set_yticks(np.arange(len(y_ticklabels)))
ax.set_yticklabels(y_ticklabels)
plt.setp(ax.get_yticklabels(), rotation=45, ha="right",
          rotation_mode="anchor")
#ax.xlabel([1,2,3,43,5,6])
#plt.set_xticks
# vals = np.around(df.values, 2)
# print(vals)
# norm = plt.Normalize(vals.min()-1, vals.max()+1)
# normal = pyplot.Normalize(vals.min()-1, vals.max()+1)
# colours = plt.cm.hot(normal(vals))

# print(vals)
# the_table=plt.table(cellText=vals, rowLabels=df.index, colLabels=df.columns, 
#                     colWidths = [0.03]*vals.shape[1], loc='center', 
#                     cellColours=colours)

plt.show()

print(df)