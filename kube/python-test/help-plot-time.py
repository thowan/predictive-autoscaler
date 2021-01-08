import matplotlib.pyplot as plt
import numpy as np

# Import seaborn
import seaborn as sns

# Apply the default theme
sns.set_theme()

# vpa = [298.550462962963, 288.31018518518516, 265.06805555555553, 262.03657407407417, 260.00555555555553, 256.9129629629629, 253.58287037037047, 262.60972222222216, 280.3064814814814, 318.4583333333333]
# hw = [272.1401870095615, 290.03836994425757, 242.6604834597534, 256.67049581112684, 218.51360962491552, 215.7767658466193, 204.0933517520074, 175.23471204200723, 190.2973505034333, 184.3440178036881]
# vpa4 = [298.550462962963, 288.31018518518516, 265.06805555555553, 262.03657407407417, 260.00555555555553, 256.9129629629629, 253.58287037037047, 262.60972222222216, 280.3064814814814, 318.4583333333333]
# hw4 = [337.8771785795001, 335.1680753235196, 291.44033561176315, 286.6401064748239, 218.51360962491552, 245.60304639324843, 236.40292679977054, 232.50840407162943, 190.2973505034333, 203.90035044666928]

# vpa = [298.550462962963, 288.31018518518516, 265.06805555555553, 262.03657407407417, 260.00555555555553, 256.9129629629629, 253.58287037037047, 262.60972222222216, 280.3064814814814, 318.4583333333333]
# hw = [272.1401870095615, 290.03836994425757, 242.6604834597534, 256.67049581112684, 218.51360962491552, 215.7767658466193, 204.0933517520074, 175.23471204200723, 190.2973505034333, 184.3440178036881]
#perc above vpa:  
vpa = [6.419999999999999, 6.6000000000000005, 6.94, 6.6000000000000005, 6.08, 4.859999999999999, 4.859999999999999, 3.82, 1.5599999999999998, 0.0]
#perc above hw:  
hw = [9.9, 8.68, 7.64, 4.34, 4.17, 2.78, 1.7399999999999998, 0.52, 0.0, 0.0]
# avg above:  [119.83333333333324, 112.51851851851852, 99.43793103448276, 88.06153846153838, 64.72307692307687, 52.57272727272723, 43.314285714285624, 27.35882352941172, 9.585714285714241, 0]
# avg above hw:  [125.42573949077862, 87.67239921448201, 93.52617054020713, 91.66104888834515, 77.83576286141368, 54.156553266813745, 43.26586519074864, 28.347224146388857, 0, 0]

x = np.linspace(0.1, 1,dtype = float, num=10)

fig = plt.figure(1)
ax1 = fig.add_subplot(1,1,1)


ax1.plot(x, hw, 'ro-', linewidth=4, label='Predictive autoscaler')
ax1.plot(x, vpa, 'yo-', linewidth=4, label='Estimated VPA target')


# ax1.plot(x, vpa4, 'go-', linewidth=4, label='VPA4 slack')
# ax1.plot(x, hw4, 'yo-', linewidth=4, label='HW4 slack')

t = "CPU usage above requested (observations)"
fig.suptitle(t, fontsize=30)

#fig.text(0.5, 0.01, txt, wrap=True, horizontalalignment='center', size=20)
ax1.tick_params(axis="x", labelsize=20) 
ax1.tick_params(axis="y", labelsize=20) 

ax1.legend(loc='lower center', bbox_to_anchor=(0.5, -0.30), fancybox=True, shadow=True, ncol=5, fontsize=24)
ax1.set_xlabel('alpha', fontsize=20)
ax1.set_ylabel('Observations (%)', fontsize=20)
fig.subplots_adjust(left=0.1, bottom=0.2, right=None, top=None, wspace=None, hspace=None)


fig.set_size_inches(15,8)
fig.savefig("./results/help_timeabove.png",bbox_inches='tight')