from kubernetes import client, config, watch
from kubernetes.client.rest import ApiException

from pprint import pprint
import time
import json
import math
import argparse

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style
import numpy as np
from statsmodels.tsa.api import ExponentialSmoothing
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from sklearn.metrics import mean_squared_error
from pandas import read_csv
from matplotlib import pyplot
import threading

import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning
warnings.simplefilter('ignore', ConvergenceWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)

# Import seaborn
import seaborn as sns

#--------------------------------------------------------------

# Apply the default theme
sns.set_theme()
# K8s config 
config.load_kube_config()
api_client = client.ApiClient()
#api_client = None

parser = argparse.ArgumentParser()

parser.add_argument("-d", "--dep_name", help="Deployment name", default="nginx-deployment")
parser.add_argument("-c", "--cont_name", help="Container name", default="nginx")
parser.add_argument("-v", "--vpa_name", help="VPA name", default="my-rec-vpa")



args = parser.parse_args()

# Plots 
fig1 = plt.figure(1)
fig2 = plt.figure(2)
ax1 = fig1.add_subplot(1,1,1)
ax2 = fig2.add_subplot(1,1,1)

# Record VPA or not
plotVPA = True

# Use LSTM or HW
use_lstm = True
print("use_lstm:", use_lstm)

vpa_x = []
vpa_targets = []
vpa_lowers = []
vpa_uppers = []

cpu_x = []
cpu_usages = []
cpu_requests = []

pred_x = []
pred_targets = []
pred_lowers = []
pred_uppers = []

cpu_slacks = []
vpa_slacks = []



def plot_slack():
    global fig2, ax2
    global cpu_slacks, vpa_slacks

    ax2.clear()
    skip = params["season_len"]*2
    #print(cpu_x[skip:])
    #print(cpu_slacks)
    ax2.plot(cpu_x[skip:], cpu_slacks, 'b--', linewidth=2, label='CPU slack')
    ax2.plot(cpu_x[skip:], vpa_slacks, 'g-', linewidth=2, label='VPA slack')
    ax2.legend(loc='lower center', bbox_to_anchor=(0.5, -0.30), fancybox=True, shadow=True, ncol=6, fontsize=15)
    ax2.set_xlabel('Time (s)', fontsize=20)
    ax2.set_ylabel('CPU (millicores)', fontsize=20)
   
        
        
# Plot the main graph, do not show
# VPA target, CPU requests/usage, LSTM bounds
def plot_main():
    global fig1, ax1
    global vpa_x, vpa_targets, cpu_x, cpu_usages, vpa_lowers, vpa_uppers, cpu_requests, pred_x, pred_targets, pred_lowers, pred_uppers

    ax1.clear()
    
    ax1.plot(vpa_x, vpa_targets, 'g--', linewidth=1,label='VPA target')
    ax1.plot(pred_x, pred_targets, 'r-', linewidth=2,label='Prediction target')
    ax1.fill_between(pred_x, pred_lowers, pred_uppers, facecolor='red', alpha=0.3, label="Prediction bounds")  
    ax1.plot(cpu_x, cpu_requests, 'b--', linewidth=2, label='CPU requested')
    ax1.plot(cpu_x, cpu_usages, 'y-', linewidth=1,label='CPU usage')
    ax1.legend(loc='lower center', bbox_to_anchor=(0.5, -0.30), fancybox=True, shadow=True, ncol=6, fontsize=15)
    ax1.set_xlabel('Observation', fontsize=20)
    ax1.set_ylabel('CPU (millicores)', fontsize=20)
    
        


def main():
    # Configs can be set in Configuration class directly or using helper
    # utility. If no argument provided, the config will be loaded from
    # default location.
    global api_client
    global vpa_x, vpa_targets, cpu_x, cpu_usages, vpa_lowers, vpa_uppers, cpu_requests, pred_x, pred_targets, pred_lowers, pred_uppers, cooldown
    global plotVPA 
    global data
    global fig1, fig2, ax1, ax2
    plotVPA = True

    # Fast initialize-------------------------------------------------------
    # np.random.seed(13)
    # series = create_sin_noise(A=300, D=200, per=params["season_len"], total_len=2*params["season_len"])
    # cpu_usages = series.tolist()
    # cpu_usages = [70.945423, 65.442296, 64.409863, 119.784497, 135.455987, 132.949178, 136.771872, 175.153478, 185.904079, 183.262895, 192.68772, 227.425753, 254.816212, 254.478113, 260.506071, 303.346596, 308.617227, 299.22515, 305.235282, 311.828772, 337.069025, 341.031472, 343.669081, 328.119776, 338.280342, 344.441163, 344.441163, 335.140099, 350.777963, 349.91699, 340.089565, 377.237572, 374.370458, 376.32487, 361.708703, 406.344532, 405.11971, 405.017317, 408.076594, 408.210802, 406.260407, 406.374084, 407.719058, 406.356785, 406.362214, 406.362214, 404.446423, 387.809218, 408.078899, 408.078899, 372.243456, 372.243456, 376.154602, 377.50961, 376.743622, 349.482621, 349.482621, 347.499424, 347.499424, 336.362507, 345.883619, 337.922989, 341.92927, 331.631114, 337.574917, 332.435077, 332.435077, 325.964683, 307.368739, 311.580147, 301.298436, 252.496212, 270.463215, 256.371774, 265.598395, 205.370252, 194.007955, 190.262442, 195.444643, 139.251378, 142.869633, 135.796605, 112.902592, 66.472793, 66.472793, 66.193411, 67.865584, 66.77803, 66.77803, 70.657887, 67.437995, 68.226009, 67.539034, 66.349509, 64.506136, 66.219697, 67.510551, 67.991665, 67.068337, 66.33974, 62.580748, 65.642412, 66.541548, 70.586391, 64.729692, 67.637338, 68.881708, 67.422404, 64.984688, 65.97749, 64.506194, 69.634008, 68.147372, 66.590158, 67.052963, 67.052963, 71.358626, 68.882216, 69.519414, 69.519414, 70.228999, 74.246052, 68.462505, 73.169358, 70.098157, 65.387362, 72.08145, 71.065703, 67.787508, 67.787508, 66.458999, 68.932161, 68.626819, 66.414441, 69.212387, 74.520871, 64.554352, 64.477423, 69.332949, 68.953759, 71.484296, 64.172944, 62.394221, 70.175386, 68.779043, 66.403721, 67.570241, 144.856782, 144.856782, 143.247413, 140.610964, 167.288596, 194.501547, 190.03789, 189.772435, 215.651063, 265.392033, 251.205724, 269.74442, 283.230117, 283.230117, 307.84315, 306.034056, 316.582364, 338.444161, 346.708794, 337.349865, 337.349865, 344.826399, 343.451073, 344.821114, 349.517567, 351.68738, 347.484378, 337.902327, 376.243902, 378.518296, 377.202081, 377.202081, 407.452107, 407.452107, 405.998908, 404.946085, 392.102185, 408.492917, 407.493804, 408.369267, 407.863674, 406.321005, 409.703717, 391.433347, 406.054267, 406.771528, 404.792209, 386.355485, 377.29304, 377.407228, 377.480392, 379.254246, 344.950444, 348.037006, 352.364188, 348.345395, 323.045971, 339.935917, 347.251705, 343.856943, 298.840484, 331.755164, 343.582499, 320.213632, 300.478595, 307.248907, 307.812206, 291.884867, 261.702297, 259.051078, 262.933378, 256.393071, 182.59464, 190.104381, 189.615586, 184.850591, 146.253126, 141.016502, 142.792523, 131.8115, 131.8115, 68.825431, 66.605583, 67.565538, 63.642538, 66.383205, 64.75593, 67.763187, 70.798345, 66.71584, 66.490961, 66.544909, 71.078104, 63.922763, 67.359739, 67.359739, 68.505196, 67.824338, 67.923622, 65.67874, 68.718988, 68.718988, 69.227808, 68.567626, 66.236974, 66.717442, 70.859004, 64.59916, 71.125389, 71.233049, 71.233049, 67.978006, 69.599998, 65.841129, 65.242505, 70.706964, 66.175486, 65.676652, 68.72556, 65.771229, 68.517979, 70.358395, 66.507317, 66.69204, 67.858524, 66.551129, 64.101733, 71.257589, 71.290914, 68.487113, 68.487113, 69.81107, 69.81107, 71.721427, 65.040494, 66.710092, 68.665638, 66.996037, 62.962659, 62.962659, 69.487323, 67.348318, 64.879687, 64.16536]
    
    # cpu_usages = cpu_usages[:288]
    # cpu_usages = cpu_usages 
    # print(len(cpu_usages))
    # cpu_requests = [700]*len(cpu_usages)
    # cpu_x = range(len(cpu_usages))

    # pred_targets = [np.nan]*len(cpu_usages)
    # pred_lowers = [np.nan]*len(cpu_usages)
    # pred_uppers = [np.nan]*len(cpu_usages)
    # pred_x = range(len(cpu_usages))
    
    # vpa_targets = [np.nan]*len(cpu_usages)
    # vpa_lowers = [np.nan]*len(cpu_usages)
    # vpa_uppers = [np.nan]*len(cpu_usages)
    # vpa_x = range(len(cpu_usages))
    #--------------------------------------------------------------------

# Plot setups ------------------------------------------
    # Set plot title, legend, labels
    fig1.suptitle('nginx pod metrics', fontsize=23)
    fig1.subplots_adjust(left=0.1, bottom=0.2, right=None, top=None, wspace=None, hspace=None)

    fig2.suptitle('Slack', fontsize=23)
    fig2.subplots_adjust(left=0.1, bottom=0.2, right=None, top=None, wspace=None, hspace=None)

    # Plot settings
    ax1.tick_params(axis="x", labelsize=20) 
    ax1.tick_params(axis="y", labelsize=20) 

    ax2.tick_params(axis="x", labelsize=20) 
    ax2.tick_params(axis="y", labelsize=20) 
    
# ---------------------------------------------------------------------

    #cooldown = params["rescale_cooldown"]

    # ax1.set_xlim(left=params["season_len"]*2) TODO
    # ax2.set_xlim(left=params["season_len"]*2) TODO
    fig1.set_size_inches(15,8)
    fig2.set_size_inches(15,8)

    


    plot_main()
    plot_slack()
    
    fig1.savefig("./main"+ args.dep_name +str(len(pred_targets))+".png",bbox_inches='tight')
    fig2.savefig("./slack"+ args.dep_name +str(len(pred_targets))+".png",bbox_inches='tight')



    
    
    
if __name__ == '__main__':
    main()
