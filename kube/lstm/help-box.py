import matplotlib.pyplot as plt
import numpy as np

data_a = [[1,2,35,2,3,5,1,2,7,6,8], [2,7,2,2,5], [7,2,5]]
data_b = [[6,4,2], [], [2,3,5,1]]
data_c = [[6,4,2], [1,2,5,3,2], [2,3,5,1]]

ticks = ['A', 'B', 'C']

def set_box_color(bp, color):
    plt.setp(bp['boxes'], color=color)
    plt.setp(bp['whiskers'], color=color)
    plt.setp(bp['caps'], color=color)
    plt.setp(bp['medians'], color=color)

plt.figure()

bpl = plt.boxplot(data_a, positions=np.array(range(len(data_a)))*2.0-0.6, sym='+', widths=0.5, whis=(0, 100), whiskerprops = dict(linestyle='--', linewidth=2))
bpr = plt.boxplot(data_b, positions=np.array(range(len(data_b)))*2.0+0.0, sym='+', widths=0.5, whis=(0, 100), whiskerprops = dict(linestyle='--', linewidth=2))
bpw = plt.boxplot(data_c, positions=np.array(range(len(data_c)))*2.0+0.6, sym='+', widths=0.5, whis=(0, 100), whiskerprops = dict(linestyle='--', linewidth=2))
set_box_color(bpl, 'red') # colors are from http://colorbrewer2.org/
set_box_color(bpr, 'blue')
set_box_color(bpw, 'grey')

# draw temporary red and blue lines and use them to create a legend
plt.plot([], c='red', label='Apples')
plt.plot([], c='blue', label='Oranges')
plt.plot([], c='grey', label='Oranges')
plt.legend()

plt.xticks(range(0, len(ticks) * 2, 2), ticks)

plt.tight_layout()
plt.savefig('boxcompare2.png')