
from pprint import pprint
import json
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style
import numpy as np
from statsmodels.tsa.api import ExponentialSmoothing
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from sklearn.metrics import mean_squared_error
import math
import argparse
parser = argparse.ArgumentParser()

parser.add_argument("-a", "--alpha", help="Noise alpha", default=1)
parser.add_argument("-s", "--std", help="Standard deviation noise", default=100)
args = parser.parse_args()

alpha = float(args.alpha)
std = int(args.std)

import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning
warnings.simplefilter('ignore', ConvergenceWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)

# Import seaborn
import seaborn as sns

# Apply the default theme
sns.set_theme()

#style.use('fivethirtyeight')
fig = plt.figure(1)
fig2 = plt.figure(2)
ax1 = fig.add_subplot(1,1,1)
ax2 = fig2.add_subplot(1,1,1)

fig.subplots_adjust(left=0.1, bottom=0.2, right=None, top=None, wspace=None, hspace=None)


# scale_d_b = [100,125,150,175,200]
# scale_u_b = [25,50, 75, 100,150,200]
# #scale stable: only scale when prediction has been stable for x steps
# scale_u_s = [3,4,5,6]
# scale_d_s = [3,4,5,6]
# #scale stable range: define what range is stable
# stable_range = [25, 50, 75, 100]

scale_d_b = np.linspace(50, 300,dtype = int, num=8)
scale_u_b = np.linspace(50, 300,dtype = int, num=8)


def get_best_params(series):
    global params
    global scale_d_b, scale_u_b
    error = np.inf
    best_params = None
    for a in scale_d_b:
        for b in scale_u_b:

                            params["scale_down_buffer"] = a
                            params["scale_up_buffer"] = b

                            rescale_counter = 0
                            scaleup = 0
                            downscale = 0
                            CPU_request = 500
                            i = 0
                            yrequest_temp = []
                            
                            while i < len(series):
                                if i > 10:

                                    p = series[i]
                                    if CPU_request - p > params["scale_down_buffer"] and scaleup > params["scaleup_count"]  and downscale > params["scaledown_count"]:
                                        #print("CPU request wasted")
                                        CPU_request = p + params["rescale_buffer"]
                                        rescale_counter += 1
                                        downscale = 0
                                    elif p - CPU_request > params["scale_up_buffer"] and scaleup > params["scaleup_count"]  and downscale > params["scaledown_count"]: 
                                        
                                        #print("CPU request too low")
                                        CPU_request = p + params["rescale_buffer"]
                                        rescale_counter += 1
                                        scaleup = 0
                                    scaleup += 1
                                    downscale += 1
                            
                                yrequest_temp.append(CPU_request)
                                i += 1 
                            # Weighted MSE where CPU usage is higher than request is weighted x times more
                            sub = np.subtract(series,yrequest_temp)
                            for v in range(len(sub)):
                                if sub[v] > 0:
                                    sub[v] = sub[v] * 10
                            MSE = np.square(sub).mean() 
                            
                            
                            if MSE < error: 

                                error = MSE
                                best_params = params.copy()
                                
                               
    print(error)
    print(best_params)
    return best_params

s_len = 144
params = {
    "window_future": 15, #HW
    "window_past": 1, #HW
    "HW_percentile": 95, #HW
    "season_len": s_len, #HW
    "history_len": 3*s_len, #HW
    "rescale_buffer": 100, # FIX
    "scaleup_count": 15, #FIX
    "scaledown_count": 15, #FIX
    "scale_down_buffer": 100,
    "scale_up_buffer":50
}

avgslack_list_vpa = []
avgslack_list_hw = []

timeabove_vpa = []
timeabove_hw = []

avgabove_vpa = []
avgabove_hw = []


def main():

    global s_len
    global params
    global alpha, std
    # 32
    # series = [0, 0, 167.729611, 335.148582, 364.596215, 359.888837, 347.909866, 357.743546, 323.415595, 330.021396, 329.90949, 279.832067, 354.451123, 338.147328, 351.322429, 350.461969, 347.813949, 329.700445, 60.152443, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 178.347119, 345.13477, 352.663322, 352.663322, 278.647535, 331.747251, 299.422929, 268.97595, 241.059516, 275.006761, 297.038177, 361.275941, 334.665032, 343.031918, 328.55202, 328.55202, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 55.833993, 331.117702, 327.047275, 339.017647, 346.807573, 345.308319, 321.528641, 353.778592, 347.427462, 303.156126, 129.78742, 345.63814, 345.63814, 310.737877, 288.766096, 342.381463, 315.305435, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 254.422417, 344.3186, 363.962895, 330.931348, 306.566837, 324.957286, 325.987237, 351.779408, 339.805264, 244.490014, 356.401872, 319.846219, 316.824208, 334.071823, 175.895969, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 299.607382, 299.607382, 354.57729, 305.574189, 353.877201, 315.48062, 355.087342, 339.048099, 345.753183, 287.981536, 344.818499, 333.110684, 351.420975, 304.503423, 334.560932, 351.794277, 119.989534, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 295.094609, 294.127069, 321.30613, 364.542628, 366.449812, 364.855091, 339.009618, 366.451555, 237.247514, 333.862215, 367.010805, 357.351747, 357.351747, 367.250366, 339.561494, 231.407897, 231.407897, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 385.967946, 363.3489, 328.912835, 349.016026, 310.531102, 332.092053, 339.425927, 341.856589, 329.008107, 337.010056, 360.535854, 299.719519, 299.719519, 348.528156, 313.176012, 164.300734, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    # 106
    #series = [8.0, 19.6, 25.2, 41.8, 37.4, 40.0, 75.6, 77.2, 53.8, 80.4, 96.0, 91.6, 100.2, 116.8, 118.4, 111.0, 117.6, 108.2, 134.8, 130.4, 129.0, 133.6, 188.2, 170.8, 197.4, 186.0, 192.6, 227.2, 212.8, 220.4, 236.0, 267.6, 247.2, 248.8, 303.4, 272.0, 337.6, 309.2, 318.8, 307.4, 271.0, 281.6, 286.2, 323.8, 320.4, 324.0, 317.6, 328.2, 319.8, 319.4, 316.0, 307.6, 341.2, 298.8, 244.4, 244.0, 266.6, 214.2, 185.8, 175.4, 189.0, 161.6, 123.19999999999999, 130.8, 130.4, 120.0, 83.6, 101.19999999999999, 91.8, 96.4, 45.0, 42.6, 48.199999999999996, 37.8, 46.4, 42.0, 41.6, 44.199999999999996, 45.8, 47.4, 43.0, 52.6, 51.199999999999996, 42.8, 57.4, 60.0, 56.6, 52.199999999999996, 47.8, 58.4, 52.0, 60.6, 56.199999999999996, 59.8, 57.4, 60.0, 57.599999999999994, 66.19999999999999, 58.8, 61.4, 69.0, 54.599999999999994, 55.199999999999996, 65.8, 57.4, 72.0, 61.599999999999994, 77.2, 92.8, 102.39999999999999, 101.0, 112.6, 130.2, 151.8, 116.39999999999999, 146.0, 163.6, 147.2, 162.8, 184.39999999999998, 182.0, 167.6, 175.2, 179.8, 193.39999999999998, 198.0, 203.6, 203.2, 250.8, 240.39999999999998, 246.0, 259.6, 253.2, 291.8, 265.4, 280.0, 301.6, 328.2, 316.8, 316.4, 361.0, 336.6, 396.2, 369.8, 396.4, 377.0, 335.6, 358.2, 350.8, 375.4, 386.0, 398.6, 384.2, 383.8, 390.4, 381.0, 390.6, 364.2, 401.8, 365.4, 305.0, 314.6, 334.2, 276.8, 246.39999999999998, 245.0, 250.6, 219.2, 196.8, 204.39999999999998, 198.0, 187.6, 159.2, 158.8, 154.39999999999998, 163.0, 101.6, 103.2, 104.8, 107.39999999999999, 107.0, 102.6, 106.2, 98.8, 113.39999999999999, 103.0, 106.6, 112.2, 109.8, 120.39999999999999, 108.0, 115.6, 115.19999999999999, 110.8, 118.39999999999999, 117.0, 120.6, 122.19999999999999, 128.8, 125.39999999999999, 119.0, 119.6, 126.19999999999999, 124.8, 125.39999999999999, 115.0, 123.6, 121.19999999999999, 125.8, 125.39999999999999, 130.0, 127.6, 130.2, 141.8, 142.4, 156.0, 154.6, 173.2, 208.79999999999998, 205.4, 177.0, 203.6, 220.2, 213.79999999999998, 215.4, 244.0, 244.6, 238.2, 234.79999999999998, 235.4, 254.0, 259.6, 266.2, 259.79999999999995, 305.4, 306.0, 320.6, 310.2, 321.79999999999995, 358.4, 329.0, 338.6, 362.2, 391.79999999999995, 377.4, 377.0, 438.6, 404.2, 454.79999999999995, 422.4, 451.0, 432.6, 404.2, 413.79999999999995, 425.4, 444.0, 446.6, 446.2, 442.79999999999995, 455.4, 449.0, 444.6, 445.2, 432.79999999999995, 461.4, 433.0, 373.6, 369.2, 391.79999999999995, 340.4, 318.0, 290.6, 329.2, 292.79999999999995, 251.4, 265.0, 259.6, 248.2, 208.79999999999998, 236.4, 223.0, 228.6, 166.2, 176.79999999999998, 171.4, 174.0, 166.6, 178.2, 175.79999999999998, 179.4, 171.0, 172.6, 186.2, 170.79999999999998, 176.4, 182.0, 177.6, 180.2, 177.79999999999998, 180.4, 174.0, 188.6, 181.2, 178.79999999999998, 185.4, 181.0, 177.6, 178.2, 187.79999999999998, 185.4, 186.0, 191.6, 182.2, 184.79999999999998, 187.4, 189.0, 189.6, 189.2, 190.79999999999998, 207.4, 207.0, 217.6, 220.2, 230.79999999999998, 255.4, 268.0, 244.6, 280.2, 282.79999999999995, 280.4, 291.0, 301.6, 311.2, 294.79999999999995, 287.4, 304.0, 315.6, 328.2, 329.79999999999995, 329.4, 380.0, 363.6, 373.2, 378.79999999999995, 381.4, 417.0, 393.6, 405.2, 428.79999999999995, 458.4, 439.0, 442.6, 503.2, 458.79999999999995, 519.4, 491.0, 502.6, 497.2, 464.79999999999995, 482.4, 489.0, 504.6, 518.2, 518.8, 514.4, 516.0, 517.6, 509.2, 510.79999999999995, 491.4, 536.0, 490.6, 439.2, 430.79999999999995, 457.4, 406.0, 382.6, 360.2, 381.79999999999995, 358.4, 310.0, 332.6, 309.2, 307.79999999999995, 282.4, 281.0, 282.6, 279.2, 240.79999999999998, 233.39999999999998, 236.0, 233.6, 234.2, 227.79999999999998, 235.39999999999998, 234.0, 241.6, 243.2, 245.79999999999998, 243.39999999999998, 238.0, 235.6, 244.2, 242.79999999999998, 234.39999999999998, 244.0, 247.6, 244.2, 244.79999999999998, 241.39999999999998, 239.0, 250.6, 246.2, 246.79999999999998, 256.4, 252.0, 250.6, 250.2, 257.79999999999995, 259.4, 256.0, 254.6, 243.2, 253.79999999999998, 254.39999999999998, 272.0, 278.6, 271.2, 278.8, 300.4, 323.0, 325.59999999999997, 313.2, 333.8, 345.4, 341.0, 354.59999999999997, 370.2, 374.8, 364.4, 363.0, 358.59999999999997, 387.2, 389.8, 400.4, 384.0, 440.59999999999997, 424.2, 436.8, 447.4, 442.0, 486.59999999999997, 461.2, 468.8, 491.4, 530.0, 503.59999999999997, 502.2, 560.8, 525.4, 583.0, 552.5999999999999, 573.2, 563.8, 528.4, 555.0, 543.5999999999999, 564.2, 565.8, 581.4, 568.0, 576.5999999999999, 580.2, 570.8, 584.4, 566.0, 604.5999999999999, 554.2, 503.8, 498.4, 524.0, 471.59999999999997, 439.2, 429.8, 451.4, 405.0, 389.59999999999997, 397.2, 383.8, 377.4, 341.0, 351.59999999999997, 356.2, 356.8, 309.4, 297.0, 294.59999999999997, 296.2, 293.8, 299.4, 301.0, 300.59999999999997, 306.2, 300.8, 301.4, 301.0, 295.59999999999997, 297.2, 299.8, 311.4, 310.0, 312.59999999999997, 301.2, 298.8, 305.4, 309.0, 311.59999999999997, 315.2, 311.8, 319.4, 311.0, 308.59999999999997, 305.2, 309.8, 315.4, 311.0, 319.59999999999997, 323.2, 305.8, 321.4, 307.0, 337.59999999999997, 344.2, 349.8, 339.4, 360.0, 387.59999999999997, 395.2, 371.8, 395.4, 415.0, 396.59999999999997, 413.2, 441.8, 432.4, 425.0, 433.59999999999997, 415.2, 457.8, 449.4, 461.0, 448.59999999999997, 507.2, 491.8, 500.4, 518.0, 498.59999999999997, 546.2, 515.8, 527.4, 563.0, 583.5999999999999, 570.2, 567.8, 619.4, 590.0, 651.5999999999999, 624.2, 638.8, 629.4, 596.0, 597.5999999999999, 612.2, 635.8, 643.4, 642.0, 637.5999999999999, 637.2, 644.8, 644.4, 629.0, 620.5999999999999, 656.2, 626.8, 559.4, 557.0, 583.5999999999999, 532.2, 504.8, 480.4, 513.0, 483.59999999999997, 444.2, 451.8, 449.4, 442.0, 407.59999999999997, 415.2, 409.8, 413.4, 353.0, 362.59999999999997, 357.2, 362.8, 355.4, 361.0, 363.59999999999997, 356.2, 364.8, 364.4, 358.0, 360.59999999999997, 370.2, 371.8, 368.4, 371.0, 370.59999999999997, 371.2, 376.8, 364.4, 375.0, 371.59999999999997, 372.2, 372.8, 373.4, 369.0, 375.59999999999997, 384.2, 376.8, 377.4, 375.0, 378.59999999999997, 381.2, 378.8, 378.4, 383.0]
    #68
    #series = [0, 4.130647, 4.130647, 21.795476, 21.795476, 23.496572, 45.288558, 49.678249, 47.942691, 50.51753, 82.670239, 74.161687, 66.922517, 81.541339, 91.649894, 88.68216, 92.902832, 102.56744, 114.015683, 120.178035, 125.270133, 126.203859, 126.203859, 144.585278, 144.585278, 145.223039, 157.521133, 187.857593, 185.830969, 180.316585, 201.03976, 229.665043, 217.553871, 193.072984, 201.853637, 264.240609, 258.400961, 224.44851, 263.330087, 244.755514, 309.953892, 290.98333, 243.207998, 252.251806, 257.077758, 253.184756, 254.75258, 234.109016, 271.299451, 287.678049, 293.678123, 292.652498, 269.487145, 244.65133, 212.643868, 208.828186, 230.871284, 229.955484, 164.450233, 139.415916, 141.457476, 141.457476, 144.376759, 95.872258, 105.457934, 93.476844, 91.500613, 53.172357, 49.207942, 49.207942, 45.391121, 24.950716, 21.77242, 20.493689, 41.972557, 41.972557, 51.425459, 48.696488, 48.696488, 67.805954, 67.805954, 64.917615, 66.484741, 91.386022, 86.236908, 92.996677, 110.001567, 110.001567, 128.146265, 128.146265, 113.019719, 160.064888, 147.927269, 138.954834, 149.033216, 160.411278, 173.346572, 156.714106, 163.902845, 218.936136, 206.200581, 234.702473, 195.599504, 254.203787, 255.270574, 241.635038, 278.682283, 254.269798, 254.269798, 298.132481, 268.953167, 286.827668, 293.509377, 260.763596, 280.347061, 262.95966, 268.925114, 277.362464, 272.423642, 238.743585, 263.402121, 244.968816, 210.744162, 191.960429, 207.855624, 191.021203, 147.365126, 131.985264, 132.142178, 153.642293, 104.940895, 88.831473, 99.308789, 95.716033, 45.215381, 55.826481, 47.40111, 49.195336, 22.13606, 25.132204, 23.464134, 27.1553, 46.550016, 47.115049, 49.574711, 49.733327, 66.602825, 66.189302, 68.25406, 79.320586, 88.315035, 88.315035, 88.18508, 83.307475, 111.593798, 107.948624, 124.947697, 127.241637, 151.997716, 145.095548, 144.679165, 140.803284, 176.396611, 187.239817, 213.512966, 168.630573, 171.856259, 219.521003, 219.879085, 222.052578, 232.011038, 272.152364, 252.967104, 255.628069, 242.339552, 242.339552, 294.632122, 271.369909, 254.146697, 282.735395, 282.331249, 332.843197, 316.305821, 270.919565, 232.885322, 275.679821, 272.553414, 290.737643, 268.009808, 251.77822, 248.729373, 200.676566, 201.800294, 200.661266, 149.59002, 131.913721, 131.913721, 133.470922, 109.673459, 86.416288, 92.218414, 85.558598, 85.558598, 41.05631, 51.499297, 51.499297, 50.499143, 22.384599, 22.384599, 27.614366, 31.777327, 51.434632, 51.889931, 56.417026, 50.761914, 0, 70.46621, 78.757378, 87.498347, 87.498347, 96.691401, 88.664483, 92.649186, 136.746531, 108.557585, 115.084785, 108.187099, 156.135403, 0, 146.822322, 173.080605, 173.080605, 171.177058, 202.478611, 209.96944, 186.890641, 210.167533, 210.167533, 228.915251, 242.925915, 239.780767, 278.992505, 253.27744, 278.297851, 302.253467, 279.893985, 267.856096, 206.839185, 312.205229, 236.689835, 283.199083, 249.919387, 271.090235, 277.487245, 278.005443, 257.907662, 273.048528, 273.048528, 226.978035, 200.59869, 215.648673, 230.408407, 152.206656, 145.400805, 144.847866, 164.770695, 0, 94.306024, 109.800772, 117.442172, 46.08921, 58.146354, 52.429386, 35.120619, 35.120619, 22.398282, 23.145454, 30.623938, 17.774605, 48.816362, 49.630316, 48.4163, 40.41872, 71.273087, 75.132752, 73.017816, 84.482425, 98.591625, 109.977262, 97.737632, 97.737632, 125.813062, 117.207876, 116.334853, 142.249459, 153.841396, 153.841396, 144.327993, 203.068903, 170.033729, 0, 178.850374, 196.483811, 202.079027, 202.079027, 122.525321, 235.021111, 220.802256, 249.030811, 237.660001, 237.660001, 255.720632, 276.485315, 281.896545, 254.810923, 248.220622, 248.220622, 265.032505, 265.032505, 308.802388, 249.832755, 261.27075, 261.27075, 273.606114, 268.816815, 285.241347, 212.270873, 218.674483, 206.319564, 197.006361, 140.489142, 143.691565, 169.476384, 142.698195, 96.215021, 98.282896, 98.282896, 93.522941, 58.384311, 53.458047, 55.634864, 38.069348, 38.069348, 13.552127, 13.552127, 13.803107, 36.782155, 41.581266, 53.181803, 40.648694, 73.217269, 78.957537, 78.957537, 95.479582, 96.931658, 99.079205, 92.6748, 110.047289, 113.556464, 112.338018, 128.937575, 160.968939, 134.219611, 155.854145, 126.32466, 155.921766, 165.907261, 177.944742, 215.172845, 215.172845, 187.596659, 169.648015, 169.648015, 228.818301, 208.138296, 255.741102, 285.346045, 248.240814, 253.344009, 254.137797, 254.137797, 227.313774, 253.370308, 249.756647, 240.614934, 229.994181, 263.317234, 242.167088, 304.412854, 219.310871, 242.122396, 197.285382, 208.193644, 172.430875, 140.595641, 141.437199, 131.848149, 144.638359, 114.74188]
    
    #100
    # series = [0, 0, 9.832358, 28.327097, 28.327097, 20.425967, 31.137821, 47.968313, 42.219266, 46.91088, 51.108978, 65.915771, 66.74087, 69.197183, 69.197183, 90.908428, 95.388223, 97.754935, 97.754935, 103.591208, 124.087233, 123.032528, 118.461505, 144.360811, 144.360811, 150.446271, 161.100811, 186.935388, 176.025295, 176.736755, 177.684402, 193.794015, 208.675785, 210.646088, 225.411648, 236.102367, 222.198626, 251.714012, 244.116205, 313.243628, 256.065858, 259.379377, 243.31113, 309.681186, 282.669042, 328.764861, 286.811298, 284.675145, 298.314133, 277.968659, 266.790163, 319.188253, 275.539988, 290.018031, 303.013228, 218.992689, 196.553283, 215.893888, 187.731775, 139.016459, 149.22681, 145.435581, 151.568947, 141.098224, 141.098224, 138.907405, 138.907405, 138.386312, 170.2671, 176.710323, 163.423801, 196.123938, 224.411747, 220.543124, 198.394648, 228.534174, 254.751507, 253.050981, 253.203161, 286.580229, 285.359278, 275.994362, 275.994362, 245.162826, 257.275309, 240.316071, 240.316071, 230.502552, 190.413125, 184.669801, 198.65853, 163.950664, 141.232634, 159.091776, 154.606384, 99.854947, 104.836828, 106.581227, 95.14845, 44.087243, 46.357131, 46.526031, 47.433219, 18.577964, 27.557768, 23.691137, 23.691137, 41.764222, 46.733211, 46.733211, 46.635433, 73.795445, 72.096101, 64.458427, 86.433311, 88.018811, 86.835934, 80.479617, 114.934479, 126.381305, 132.537303, 121.456865, 125.877147, 152.229137, 150.118879, 137.386232, 171.084665, 190.55211, 184.121186, 168.184301, 211.738981, 211.353014, 213.610787, 223.221091, 257.842529, 266.738772, 289.963931, 289.963931, 267.609999, 261.384933, 261.384933, 268.895763, 276.286647, 311.832433, 295.865833, 314.832796, 258.169944, 283.977941, 309.77613, 309.77613, 346.132698, 346.132698, 275.028602, 331.229079, 310.098232, 251.21078, 222.371238, 215.003516, 226.596595, 167.057717, 146.522128, 164.071938, 150.01241, 158.460901, 153.228797, 145.434871, 183.740297, 196.7824, 157.040954, 181.159763, 215.712998, 216.282971, 213.293038, 242.67215, 232.641153, 253.021053, 245.699697, 276.919804, 276.919804, 302.033114, 283.420494, 320.279425, 320.279425, 317.700541, 254.961111, 254.961111, 286.333272, 246.052235, 216.476855, 205.852836, 205.852836, 179.342487, 153.629359, 145.931036, 126.172647, 100.63973, 100.63973, 96.766543, 104.81366, 67.039087, 52.160349, 49.227496, 42.552188, 42.552188, 23.284737, 23.992347, 24.566197, 35.230453, 44.187225, 50.036957, 47.96284, 65.35303, 67.47664, 71.938785, 67.077398, 93.971032, 93.971032, 86.556906, 114.251797, 114.251797, 119.463373, 121.011744, 136.647943, 141.950177, 156.929595, 130.695672, 165.395797, 168.898773, 168.898773, 199.558611, 218.647593, 218.647593, 216.730326, 216.730326, 247.011775, 281.023474, 234.455904, 276.96152, 244.503266, 325.532657, 245.355676, 294.387039, 260.701897, 270.43564, 292.339832, 292.339832, 222.632774, 222.632774, 240.65339, 249.823186, 249.823186, 269.945639, 269.945639, 249.836298, 281.176552, 207.833182, 190.575354, 194.910685, 214.933046, 158.450889, 146.598446, 135.118389, 129.48869, 149.548963, 148.456971, 154.398959, 154.398959, 160.319378, 186.307129, 173.252636, 175.794716, 214.120213, 214.120213, 197.051589, 241.096388, 244.963499, 272.346354, 282.985355, 282.985355, 253.218306, 260.261609, 275.922359, 276.003335, 299.114837, 267.017976, 294.811021, 260.561538, 206.190501, 180.369137, 209.957508, 163.444815, 146.228373, 147.198165, 147.683811, 147.683811, 91.003165, 90.633029, 90.633029, 83.78131, 47.47288, 44.563391, 0, 29.806889, 29.806889, 23.159171, 25.051263, 31.736623, 51.979568, 46.114922, 46.910667, 66.433719, 68.537232, 0, 67.640617, 90.234122, 95.192317, 91.10206, 100.643127, 123.272986, 116.405286, 108.353139, 108.353139, 150.150501, 0, 129.606613, 129.606613, 184.728269, 162.102068, 183.544133, 184.338753, 191.432832, 184.201487, 218.175323, 238.070912, 0, 224.111702, 224.111702, 255.387087, 252.850828, 277.287838, 273.061506, 285.995996, 275.142498, 260.955266, 266.515603, 258.86717, 291.790795, 254.398927, 223.536812, 260.196284, 260.196284, 235.508138, 208.566645, 217.318373, 182.869747, 188.656013, 140.5943, 140.5943, 145.068098, 138.462224, 138.733698, 153.935788, 155.134093, 181.030513, 161.647618, 170.324704, 175.444168, 222.068069, 202.894395, 202.894395, 0, 0, 226.817772, 191.09392, 210.729857, 202.826282, 256.463663, 259.983951, 250.97536, 277.528, 266.583935, 279.602182, 187.064896, 201.394972, 201.394972, 197.145386, 152.382089, 158.683042, 101.17683, 101.649351, 0, 87.290942, 93.754849, 47.110647, 48.551838, 50.131532, 50.131532, 24.76395, 21.796207, 21.466343, 27.256609, 44.659785, 49.379577, 51.686223, 62.072132, 70.006681, 69.14184, 72.881477, 88.906869, 99.61812, 88.494992, 0, 120.345758, 132.057785, 109.509922, 123.18301, 130.085448, 136.226128, 150.399854, 167.569791, 195.236715, 183.828099, 0, 157.640393, 196.925641, 183.795503, 193.015972, 220.37747, 236.099739, 208.067218, 253.497486, 248.628731, 233.772201, 268.568974, 0, 277.683097, 262.148132, 306.129977, 286.883963, 268.767862, 248.711281, 247.569059, 284.19884, 250.330073, 272.977081, 232.4886, 201.175211, 185.246862, 175.41846, 161.312754, 145.700849, 126.247936, 153.10872, 135.795201, 0, 140.497395, 158.480199, 161.821668, 180.263581, 183.651046, 210.913845, 183.330194, 202.158407, 228.355154, 241.285557, 198.220312, 236.774996, 0, 211.730226, 225.505776, 248.940909, 285.856068, 273.378527, 234.62643, 218.900294, 180.770953, 195.506404, 165.195304, 193.791479, 143.116294, 138.98289, 147.724234, 0, 91.409873, 96.99724, 66.462991, 41.56337, 45.276706, 44.26621, 34.762394, 23.38211, 30.465631, 26.306181, 22.641473, 54.157213, 46.420464, 64.455515, 67.61221, 72.539939, 74.601384, 77.44513, 99.177282, 90.062738, 92.472689, 96.286734, 141.331946, 128.229465, 113.686486, 145.147902, 141.601685, 171.28961, 0, 183.553868, 147.785886, 158.271762, 160.727287, 195.695114, 174.894322, 174.894322, 228.641884, 222.203622, 280.444428, 0, 261.610009, 238.476832, 270.059824, 252.877098, 277.885644, 252.735122, 236.032476, 284.086614, 244.378487, 231.180486, 231.180486, 244.080902, 226.213086, 226.213086, 196.063186, 190.550166, 183.943671, 203.718456, 163.334769, 133.059787, 141.398254, 0]
    # s1 = series[:100]
    # series = s1 + s1 + s1 +s1 +s1
    
    # series = list(range(600)) + list(reversed(range(600)))[:200] + list(reversed(list(reversed(range(600)))[:200]))

    # # print(len(series))


    # Sine wave
    A = 300
    per = s_len
    B = 2*np.pi/per
    D = 200
    sample = 7*per
    x = np.arange(sample)
    series = A*np.sin(B*x)+D
    #alpha = float(args.alpha)
    series = series * alpha
    np.random.seed(3)
    noise = np.random.normal(0,int(std),len(series))*(1-alpha)
    series = [sum(x) for x in zip(noise, series)]
    series = [int(i) for i in series]

    # noise = np.random.normal(0,20,len(series))
    # series = [sum(x) for x in zip(noise, series)]
    # series = [int(i) for i in series]
    #add = np.arange(len(series))
    #add = [x*0.3 for x in add]
    # add = len(series) * [0]
    # add[int(len(add)/1.6)] = 700
    # add[int(len(add)/1.6)+1] = 740
    # add[int(len(add)/1.6)+2] = 760
    # add[int(len(add)/1.6)+3] = 710
    # series = [sum(x) for x in zip(add, series)]
    series = [1 if i <= 0 else i for i in series]

    test_last = s_len*2

    CPU_request = 600
    yrequest = [CPU_request] * test_last
    yhat = []
    Y = []

    

    i = test_last 

    scaleup = 0
    downscale = 0
    best = False
    model = None

    rescale_counter = 0
    while i <= len(series):

        # What we have seen up until now
        series_part = series[:i]
        
        season = math.ceil((i+1)/s_len)
        history_start_season = season - (params["history_len"]/s_len)
        if history_start_season < 1:
            history_start_season = 1
        history_start = (history_start_season-1) * s_len 
        n = int(i - history_start)
        
        #model = ExponentialSmoothing(series_part[-n:], trend="add", damped=False, seasonal=None)
        #if update model
        if i % 20 == 0 or model is None:
            model = ExponentialSmoothing(series_part[-n:], trend="add", seasonal="add", seasonal_periods=s_len)
            model_fit = model.fit()
            

        
        x = model_fit.predict(start=n-params["window_past"],end=n+params["window_future"])

            
        

        
        
        
        # if i % s_len == 0:

        #print(model_fit.params_formatted)
        p = np.percentile(x, params["HW_percentile"])

        if p < 0:
            p = 0
        Y.append(p)

        # # If HW has been running for 1 season
        if len(Y) >= s_len and len(Y) % s_len == 0:

            a = Y[-params["history_len"]:]

            a = [x for x in a if not math.isnan(x)]
            
            params = get_best_params(a)
            best = True

        if best:
            CPU_request_temp = CPU_request - params["rescale_buffer"]
            if CPU_request_temp - p > params["scale_down_buffer"] and scaleup > params["scaleup_count"]  and downscale > params["scaledown_count"]:
            
                #print("CPU request wasted")
                CPU_request = p + params["rescale_buffer"]
                rescale_counter += 1
                downscale = 0
            elif p - CPU_request_temp > params["scale_up_buffer"] and scaleup > params["scaleup_count"]  and downscale > params["scaledown_count"]: 
               
                #print("CPU request too low")
                CPU_request = p + params["rescale_buffer"]
                rescale_counter += 1
                scaleup = 0
        scaleup += 1
        downscale += 1

        
        yrequest.append(CPU_request)
        i += 1
    print(rescale_counter)
    
    X = range(len(Y))
    X = [x+test_last for x in X]
    
    



    xrequest = range(len(yrequest))
    #xrequest = [x+test_last for x in xrequest]
    
    
    series_X = range(len(series))
    
    series_X3 = range(len(yhat))

    
    series_X3 = [x+test_last for x in series_X3]
    ax1.plot(X, Y, 'bo-', linewidth=4,label='HW prediction')
    ax1.plot(xrequest,yrequest, 'ro-', linewidth=4,label='CPU requested')
    ax1.plot(series_X, series, 'go-', linewidth=4,label='CPU usage')
    #ax1.plot(series_X3, yhat, 'o-', linewidth=1,label='Lib')

    #Plot estimate of VPA target
    target = np.percentile(series, 90) + 50
    vpa = [target] * len(series_X)
    ax1.plot(series_X, vpa, 'yo-', linewidth=4,label='Estimated VPA target')



    print("---AVERAGE SLACK---")
    skip = params["season_len"]
    #print(yrequest[skip:])
    yrequest.pop()

    reqs = yrequest[skip*3:]
    usages = series[skip*3:]
    vpa_short = vpa[skip*3:]

    # reqs = [350 for i in range(195)]

    avg_slack = np.average(np.subtract(reqs,usages))
    avg_slack_vpa = np.average(np.subtract(vpa[skip*3:],usages))
    
    print("HW:", avg_slack)
    print("VPA:", avg_slack_vpa)

    print("---% TIME ABOVE REQUESTED---")
    count = 0
    total = 0
    vpa_count = 0
    vpa_total = 0
    for i in range(len(usages)):
        if usages[i] > reqs[i]:
            count += 1
            total += usages[i] - reqs[i]
        if usages[i] > vpa_short[i]:
            vpa_count += 1
            vpa_total += usages[i] - vpa_short[i] 
        
    perc_above_requested = round(count/len(usages), 4)*100
    perc_above_vpa = round(vpa_count/len(usages), 4)*100
    
    print("Usage above requested: " + str(perc_above_requested) + "%")
    print("Exact: " + str(count/len(usages)))
    print("Usage above vpa target: " + str(perc_above_vpa) + "%")
    print("Exact: " + str(vpa_count/len(usages)))

    print("---AVERAGE USAGE ABOVE REQUESTED---")
    if count > 0:
        print("HW: " + str(total/count))
    if vpa_count > 0:
        print("VPA: "+ str(vpa_total/vpa_count))
    
    # ax1.set_ylim(bottom=-1)
    t = ("CPU autoscaling, alpha: " + str(round(alpha,1)))
    fig.suptitle(t, fontsize=30)
    
    #fig.text(0.5, 0.01, txt, wrap=True, horizontalalignment='center', size=20)
    ax1.tick_params(axis="x", labelsize=20) 
    ax1.tick_params(axis="y", labelsize=20) 
    # ax1.legend(loc='best', fancybox=True, shadow=True, ncol=5, fontsize=24)
    ax1.legend(loc='lower center', bbox_to_anchor=(0.5, -0.30), fancybox=True, shadow=True, ncol=5, fontsize=20)
    ax1.set_xlabel('Observations', fontsize=20)
    ax1.set_ylabel('CPU (millicores)', fontsize=20)



    # Take all (requested CPU - CPU usage)/n

    




    hw_slack = np.subtract(yrequest,series)
    #hw_slack = [0 if i < 0 else i for i in hw_slack]
    vpa_slack = np.subtract(vpa,series)
    #vpa_slack = [0 if i < 0 else i for i in vpa_slack]

    ax2.plot(series_X, hw_slack, 'ro-', linewidth=4, label='Predictive autoscaler')
    ax2.plot(series_X, vpa_slack, 'yo-', linewidth=4, label='Estimated VPA target')

    t2 = "CPU slack, alpha: " + str(round(alpha,1))
    #t2 = ("CPU slack, alpha: " + str(alpha) + "\n VPA avg slack: " + str(int(avg_slack_vpa)) + ", HW avg slack: "+ str(int(avg_slack)))
    fig2.suptitle(t2, fontsize=30)
    ax2.tick_params(axis="x", labelsize=20) 
    ax2.tick_params(axis="y", labelsize=20) 
    fig2.subplots_adjust(left=0.1, bottom=0.2, right=None, top=None, wspace=None, hspace=None)
    # ax2.legend(loc='best', fancybox=True, shadow=True, ncol=5, fontsize=24)
    ax2.legend(loc='lower center', bbox_to_anchor=(0.5, -0.30), fancybox=True, shadow=True, ncol=5, fontsize=20)
    ax2.set_xlabel('Observations', fontsize=20)
    ax2.set_ylabel('CPU (millicores)', fontsize=20)
    # ax2.set_ylim(bottom=-100)
    # ax2.set_ylim(top=505)

    #plt.show()
    #plt.savefig('fig1.png', figsize=(19.2,10.8), dpi=100)

   # manager = plt.get_current_fig_manager()
    #manager.resize(*manager.window.maxsize())
    ax1.set_xlim(left=s_len*3)
    ax2.set_xlim(left=s_len*3)
    fig.set_size_inches(15,8)
    fig2.set_size_inches(15,8)
    plt.show()

    fig.savefig("./results/scale"+str(int(alpha*10))+".png",bbox_inches='tight')
    fig2.savefig("./results/slack"+str(int(alpha*10))+".png", bbox_inches="tight")  

    ax1.clear() 
    ax2.clear() 
    
    avgslack_list_vpa.append(avg_slack_vpa)
    avgslack_list_hw.append(avg_slack)

    print("avg slack vpa: ", avgslack_list_vpa)
    print("avg slack hw: ", avgslack_list_hw)

    timeabove_vpa.append(perc_above_vpa)
    timeabove_hw.append(perc_above_requested)

    print("perc above vpa: ", timeabove_vpa)
    print("perc above hw: ", timeabove_hw)

    if vpa_count == 0:
        avgabove_vpa.append(0)
    else:
        avgabove_vpa.append(vpa_total/vpa_count)
    if count == 0:
        avgabove_hw.append(0)
    else:
        avgabove_hw.append(total/count)

    print("avg above: ", avgabove_vpa)
    print("avg above hw: ", avgabove_hw)



    
    
    
    
    

if __name__ == '__main__':
    alphas = np.linspace(0.1, 1,dtype = float, num=10)
    std = 300
    for i in range(len(alphas)):
        print(alphas[i])
        alpha = alphas[i]
        main()

    
    
