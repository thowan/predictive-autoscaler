# Python code to
# demonstrate readlines()
import matplotlib.pyplot as plt
import numpy as np
 

# Using readlines()
file1 = open('c_1.txt', 'r')
Lines = file1.readlines()

 
a = []
b = []
i = 0
start_time = 0
# Strips the newline character
for line in Lines:
    
    for word in line.split():
        if i % 2 == 0:
            #First column
            b.append(float(word))
        else:
            a.append(float(word))
        i += 1
    # if i > 801:
    #     break

a = [i * 40 for i in a]
# print(a)
#print(a)

xs = []
ys = []

interval = 600
# 1146 data points with 600 seconds interval (10min each)
# print(int((b[-1] - b[0]) / 600))
total = int((b[-1] - b[0]) / interval)
for i in range(total):
    x = i*interval + b[0]
    y = int(np.interp(x, b, a))
    xs.append(x)
    ys.append(y)
# np.interp(start_time, b, a)


plt.plot(xs, ys, '.-')
plt.xlabel('Observations (10 min interval)')
plt.ylabel('Millicores')
print(ys)


plt.savefig("d.png",bbox_inches='tight')