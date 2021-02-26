import matplotlib.pyplot as plt
import numpy as np

# Import seaborn
import seaborn as sns

# Apply the default theme
sns.set_theme()
sns.set_style("whitegrid")


# vpa = [6.419999999999999, 6.6000000000000005, 6.94, 6.6000000000000005, 6.08, 4.859999999999999, 4.859999999999999, 3.82, 1.5599999999999998, 0.0]  
# hw = [9.9, 8.68, 7.64, 4.34, 4.17, 2.78, 1.7399999999999998, 0.52, 0.0, 0.0]


# perc above vpa_target:  [7.8100000000000005, 8.16, 7.12, 6.94, 6.6000000000000005, 6.25, 5.21, 3.1199999999999997, 1.39, 0.0]
# perc above hw:  [10.24, 10.59, 9.719999999999999, 7.470000000000001, 6.6000000000000005, 3.9899999999999998, 2.6, 0.69, 0.0, 0.0]
# perc above lstm:  [5.7299999999999995, 6.25, 6.77, 4.6899999999999995, 4.51, 2.4299999999999997, 1.91, 0.52, 0.0, 0.0]

vpa = [7.8100000000000005, 8.16, 7.12, 6.94, 6.6000000000000005, 6.25, 5.21, 3.1199999999999997, 1.39, 0.0]
hw = [10.24, 10.59, 9.719999999999999, 7.470000000000001, 6.6000000000000005, 3.9899999999999998, 2.6, 0.69, 0.0, 0.0]
lstm = [5.7299999999999995, 6.25, 6.77, 4.6899999999999995, 4.51, 2.4299999999999997, 1.91, 0.52, 0.0, 0.0]
x = np.linspace(0.1, 1,dtype = float, num=10)

fig = plt.figure(1)
ax1 = fig.add_subplot(1,1,1)
fig.subplots_adjust(left=0.1, bottom=0.2, right=None, top=None, wspace=None, hspace=None)
t = "% usage above requested"
fig.suptitle(t, fontsize=23)

fig.set_size_inches(15,8)

ax1.plot(x, vpa, 'yo-', linewidth=2, label='VPA')
ax1.plot(x, hw, 'ro-', linewidth=2, label='HW')
ax1.plot(x, lstm, 'bo-', linewidth=2, label='LSTM')
plt.xticks(np.arange(min(x), max(x)+0.1, 0.1))

# ax1.plot(x, vpa4, 'go-', linewidth=4, label='VPA4 slack')
# ax1.plot(x, hw4, 'yo-', linewidth=4, label='HW4 slack')


#fig.text(0.5, 0.01, txt, wrap=True, horizontalalignment='center', size=20)
ax1.tick_params(axis="x", labelsize=20) 
ax1.tick_params(axis="y", labelsize=20) 


leg4 = ax1.legend(loc='lower center', bbox_to_anchor=(0.5, -0.30), fancybox=True, shadow=True, ncol=6, fontsize=22, labelspacing=0.1, handletextpad=0.3, columnspacing=0.4)
leg4_lines = leg4.get_lines()
plt.setp(leg4_lines, linewidth=5)

ax1.set_xlabel('Alpha', fontsize=25)
ax1.set_ylabel('Observations (%)', fontsize=25)




fig.savefig("./help_timeabove.png",bbox_inches='tight')