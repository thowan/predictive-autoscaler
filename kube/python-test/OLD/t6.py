
from pprint import pprint
import json
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style
import numpy as np
from statsmodels.tsa.api import ExponentialSmoothing
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from sklearn.metrics import mean_squared_error
import csv

style.use('fivethirtyeight')
fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)
box = ax1.get_position()
#ax1.set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.9])
plt.subplots_adjust(left=0.1, bottom=0.2, right=None, top=None, wspace=None, hspace=None)


def initial_trend(series, slen):
    sum = 0.0
    for i in range(slen):
        sum += float(series[i+slen] - series[i]) / slen
    return sum / slen

def initial_seasonal_components(series, slen):
    seasonals = {}
    season_averages = []
    n_seasons = int(len(series)/slen)
    # compute season averages
    for j in range(n_seasons):
        season_averages.append(sum(series[slen*j:slen*j+slen])/float(slen))
    # compute initial values
    for i in range(slen):
        sum_of_vals_over_avg = 0.0
        for j in range(n_seasons):
            sum_of_vals_over_avg += series[slen*j+i]-season_averages[j]
        seasonals[i] = sum_of_vals_over_avg/n_seasons
    return seasonals

def triple_exponential_smoothing(series, slen, alpha, beta, gamma, n_preds):
    result = []
    seasonals = initial_seasonal_components(series, slen)
    for i in range(len(series)+n_preds):
        if i == 0: # initial values
            smooth = series[0]
            trend = initial_trend(series, slen)
            result.append(series[0])
            continue
        if i >= len(series): # we are forecasting
            m = i - len(series) + 1
            result.append((smooth + m*trend) + seasonals[i%slen])
        else:
            val = series[i]
            last_smooth, smooth = smooth, alpha*(val-seasonals[i%slen]) + (1-alpha)*(smooth+trend)
            trend = beta * (smooth-last_smooth) + (1-beta)*trend
            seasonals[i%slen] = gamma*(val-smooth) + (1-gamma)*seasonals[i%slen]
            result.append(smooth+trend+seasonals[i%slen])
    return result


def write_data():
    # data to be written row-wise in csv fil 
    data = [[12], [4], [5,35,4,1]] 
    
    # opening the csv file in 'w+' mode 
    file = open('data.csv', 'w+', newline ='') 
    
    # writing the data into the file 
    with file:     
        write = csv.writer(file) 
        write.writerows(data) 



# Read csv file into lists
def read_data():
    with open('data.csv', 'r') as read_obj:
        # pass the file object to reader() to get the reader object
        csv_reader = csv.reader(read_obj)
        # Pass reader object to list() to get a list of lists
        list_of_rows = list(csv_reader)



    VPA_upper = list(map(float, (list_of_rows[0]))) 
    VPA_lower = list(map(float, (list_of_rows[1]))) 
    VPA_target = list(map(float, (list_of_rows[2]))) 
    requested = list(map(float, (list_of_rows[3]))) 
    usage = list(map(float, (list_of_rows[4]))) 
    HW = list(map(float, (list_of_rows[5]))) 
    
    
    

    return VPA_upper, VPA_lower, VPA_target, requested, usage, HW

scale_d_b = [75, 100,150,200,250]
scale_u_b = [75, 100,150,200,250]
#scale stable: only scale when prediction has been stable for x steps
scale_u_s = [3,4,5,6]
scale_d_s = [3,4,5,6]
#scale stable range: define what range is stable
stable_range = [25, 50, 75 , 100,150]

#PARAMETERS
params = {
    "window_future": 2, #HW
    "window_past": 0, #HW
    "HW_percentile": 95, #HW
    "season_len": s_len, #HW
    "history_len": 3*s_len, #HW
    "scale_down_buffer": 100,
    "scale_up_buffer":50,
    "scale_up_stable":1,
    "scale_down_stable":1,
    "rescale_buffer": 25, # FIX
    "rescale_max": 5,
    "scaleup_count": 10, #FIX
    "scaledown_count": 10, #FIX
    "stable_range": 50

}

def get_best_params(series):
    global params
    global scale_d_b, scale_u_b, scale_u_s, scale_d_s, stable_range
    error = np.inf
    best_params = None
    for a in scale_d_b:
        for b in scale_u_b:
            for c in scale_u_s:
                for d in scale_d_s:
                    for e in stable_range:

                            params["scale_down_buffer"] = a
                            params["scale_up_buffer"] = b
                            params["scale_up_stable"] = c
                            params["scale_down_stable"] = d
                            params["stable_range"] = e
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
                                        if np.max(series[i-params["scale_down_stable"]:i+1])-np.min(series[i-params["scale_down_stable"]:i+1]) < params["stable_range"]:
                                            #print("CPU request wasted")
                                            CPU_request = p + params["rescale_buffer"]
                                            rescale_counter += 1
                                            downscale = 0
                                    elif p - CPU_request > params["scale_up_buffer"] and scaleup > params["scaleup_count"]  and downscale > params["scaledown_count"]: 
                                        if np.max(series[i-params["scale_up_stable"]:i+1])-np.min(series[i-params["scale_up_stable"]:i+1]) < params["stable_range"]:
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
                            
                            # Maximum rescale x times 
                            if rescale_counter < 100 and MSE < error: 

                                error = MSE
                                yrequest = yrequest_temp
                                best_params = params.copy()
                                set_best = True
                                
                        
    print(error)
    print(best_params)
    return best_params

def main():


    series = [372.985744, 309.186344, 309.186344, 326.524756, 367.593372, 322.23134, 361.87763, 377.021857, 372.73047, 144.805168, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 13.981515, 363.636463, 332.690807, 282.461721, 305.749716, 313.993662, 368.340288, 385.984526, 325.853594, 339.850816, 373.752876, 364.267271, 340.255855, 371.559231, 371.559231, 335.829166, 321.199285, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 87.881462, 381.332001, 356.065631, 337.888093, 327.060371, 360.574138, 321.934988, 261.349427, 272.259273, 235.567208, 261.312735, 251.208536, 263.100596, 267.777311, 285.618235, 271.841501, 194.254993, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 211.000958, 320.915281, 295.053191, 298.637113, 306.573511, 330.394161, 330.394161, 258.848588, 295.203199, 347.430224, 381.326969, 312.843782, 314.887124, 325.559274, 325.559274, 334.064061, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3.327715, 364.428684, 340.007105, 373.612761, 369.140033, 343.785815, 365.539178, 0, 293.047772, 299.978405, 358.808088, 343.26366, 351.887304, 335.982445, 355.233386, 361.630705, 30.541678, 30.541678, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 218.151175, 326.083429, 354.382685, 354.382685, 340.911176, 379.087897, 359.642556, 393.493804]

    s_len = 32


    HW = []
                    

    params = get_best_params()
    test_last = s_len*2
  
    
    yrequest = []
    rescale_counter = 0
    scaleup = 0
    downscale = 0
    CPU_request = 500
    i = test_last
    while i <= len(series):
        if i < 3*s_len:
            n = i
        else:
            n = 3*s_len
        series_part = series[:i]

        model = ExponentialSmoothing(series_part[-params["history_len"]:], trend="add", damped=False, seasonal="add",seasonal_periods=s_len)
        
        model_fit = model.fit()
        x = model_fit.predict(start=n-params["window_past"],end=n+params["window_future"])

        p = np.percentile(x, params["HW_percentile"])
        HW.append(p)

        if i > 0:

            p = series[i]
            if CPU_request - p > params["scale_down_buffer"] and scaleup > params["scaleup_count"]  and downscale > params["scaledown_count"]:
                if np.max(series[i-params["scale_down_stable"]:i+1])-np.min(series[i-params["scale_down_stable"]:i+1]) < params["stable_range"]:
                    #print("CPU request wasted")
                    CPU_request = p + params["rescale_buffer"]
                    rescale_counter += 1
                    downscale = 0
            elif p - CPU_request > params["scale_up_buffer"] and scaleup > params["scaleup_count"]  and downscale > params["scaledown_count"]: 
                if np.max(series[i-params["scale_up_stable"]:i+1])-np.min(series[i-params["scale_up_stable"]:i+1]) < params["stable_range"]:
                    #print("CPU request too low")
                    CPU_request = p + params["rescale_buffer"]
                    rescale_counter += 1
                    scaleup = 0
            scaleup += 1
            downscale += 1
    
        yrequest.append(CPU_request)
        i += 1 

    series_X = range(len(series))

    ax1.plot(series_X, yrequest, '.-', linewidth=1,label='CPU request')
    ax1.plot(series_X, series, '.-', linewidth=1,label='CPU usage')
    ax1.plot(series_X, HW, '.-', linewidth=1,label='CPU usage')
    #ax1.plot(series_X3, yhat, '.-', linewidth=1,label='Lib')

    
    
    ax1.set_ylim(bottom=0)
    txt=best_params
    plt.figtext(0.5, 0.01, txt, wrap=True, horizontalalignment='center', fontsize=12)
    ax1.legend(loc='lower center', bbox_to_anchor=(0.5, -0.25), fancybox=True, shadow=True, ncol=5)
    plt.show()

    

if __name__ == '__main__':
    main()