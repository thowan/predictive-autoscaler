
from pprint import pprint
import json
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style
import numpy as np
from statsmodels.tsa.api import ExponentialSmoothing
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from sklearn.metrics import mean_squared_error

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

def main():

    # series = [30,21,29,31,40,48,53,47,37,39,31,29,17,9,20,24,27,35,41,38,
    #       27,31,27,26,21,13,21,18,33,35,40,36,22,24,21,20,17,14,17,19,
    #       26,29,40,31,20,24,18,26,17,9,17,21,28,32,46,33,23,28,22,27,
    #       18,8,17,21,31,34,44,38,31,30,26,32]

    
    #series = [0, 0, 0, 222.268281, 227.201073, 239.880936, 219.442175, 224.973491, 229.811911, 205.091031, 74.613006, 39.074674, 42.660607, 46.771171, 49.42565, 42.942462, 38.4036, 84.570203, 292.503893, 315.355352, 308.760204, 323.474822, 328.579046, 320.081387, 252.186329, 303.202204, 100.9927, 90.015986, 109.322007, 109.322007, 101.237994, 93.506046, 85.536672, 113.139345, 363.6933, 336.765956, 336.765956, 360.514003, 360.514003, 386.828631, 333.582584, 311.141953, 137.747034, 126.938755, 153.468333, 139.097676, 153.49441, 137.402691, 157.353422, 157.353422, 211.900311, 383.462928, 404.630162, 381.371638, 362.084191, 370.947801, 377.015265, 399.585546, 123.408264, 113.998109, 112.856803, 106.889714, 118.416439, 120.229568, 112.551647, 149.311997, 360.410137, 322.75247, 322.75247, 317.344449, 333.973616, 325.086149, 348.288513, 318.460927, 213.543097, 74.776173, 70.599969, 77.547022, 71.492227, 60.858282, 72.131921, 228.631534, 287.955998, 252.905549, 255.666318, 293.224175, 293.224175, 254.890236, 256.31281, 184.259296, 26.746213, 20.837145, 25.642178, 20.129102, 20.129102, 28.922743, 21.160708, 152.93683, 200.704473, 226.081362, 196.194592, 200.608984, 202.260495, 173.108635, 179.423445, 17.397033, 17.397033, 0, 0, 0, 0, 0, 0, 98.800742, 192.672357, 199.414745, 215.054186, 215.054186, 205.410493, 209.414565, 203.866979, 0, 0, 0, 0, 0, 0, 0, 0, 0, 176.747441, 223.099724, 194.794772, 207.239332, 210.721222, 186.568032, 202.327088, 63.881363, 0, 0, 0, 0, 0, 0, 0, 0, 203.325941, 203.325941, 188.611366, 210.255919, 200.731947, 184.48422, 174.899503, 65.183011, 0, 0, 0, 0, 0, 0, 0, 144.871317, 190.061155, 197.329181, 197.329181, 210.432742, 192.562001, 204.877713, 184.600815, 19.330818, 19.330818, 0, 0, 0, 0, 0, 0, 28.484123, 205.466999, 217.307983, 196.869224, 234.258049, 194.447863, 194.447863, 205.503725, 159.129107, 0, 0, 0, 0, 0, 0, 0, 46.571091, 211.233347, 210.528275, 209.810576, 208.042589, 201.983314, 207.541246, 103.850549, 0, 0, 0, 0, 0, 0, 0, 116.567582, 229.524659, 229.524659, 201.376278, 177.673079, 195.137758, 181.622873, 190.319366, 39.957974, 0, 0, 0, 0, 0, 0, 0, 0, 0, 208.246289, 205.835285, 210.59648, 227.831936, 207.289967, 196.390665, 65.297689, 0, 0, 0, 0, 0, 0, 0, 21.401811, 222.476651, 205.850522, 218.542979, 222.958509, 189.657957, 181.209873, 201.821104, 63.8068, 0, 0, 0, 0, 0, 0, 0, 69.248479, 210.661921, 223.802807, 241.081262, 205.547105, 243.009009, 260.567218, 260.567218, 189.80742, 0, 0, 0, 0, 0, 0, 0, 195.973302, 230.563378, 192.294364, 223.705028, 197.135762, 196.654874, 208.670291, 204.989584, 211.586295, 5.615713, 0, 0, 0, 0, 0, 0, 0, 164.63892, 228.020676, 228.020676, 190.805889, 190.805889, 222.252833, 183.420644, 202.987075, 0, 0, 0, 0, 0, 0, 0, 0, 95.945274, 213.687626, 213.687626, 213.888819, 199.37523, 201.541361, 213.847073, 122.494975, 0, 0, 0, 0, 0, 0, 0, 167.106952, 225.893407, 218.828179, 208.885838, 216.299584, 204.182913, 227.458804, 153.380786, 0, 0, 0, 0, 0, 0, 0, 32.251393, 201.108612, 235.00415, 226.748656, 235.84925, 193.113298, 226.707257, 185.68136, 176.033516, 0, 0, 0, 0, 0, 0, 0, 0.181451, 194.326056, 209.109078, 202.454246, 211.493586, 202.346181, 211.720298, 192.853697, 165.535571, 165.535571, 22.657118, 24.97674, 23.579371, 22.294711, 24.946162, 32.883981, 32.883981, 289.868464, 289.868464, 302.223187, 264.67557, 274.798801, 253.463954, 294.168544, 263.701072, 135.374528, 70.301681, 78.182321, 67.180241, 65.927087, 74.959489, 66.805405, 98.022544, 311.460364, 358.618652, 319.176362, 355.976855, 355.976855, 316.008311, 331.568436, 353.19335, 150.949045, 119.900066, 128.266793, 111.405631, 111.405631, 121.962242, 114.943344, 137.370867, 348.047149, 376.239721, 397.364878, 416.238386, 414.414955, 383.580424, 383.580424, 280.995375, 147.834219, 138.971163, 164.977046, 164.977046, 139.558758, 139.558758, 157.090652, 130.790477, 375.97726, 361.358215, 361.791625, 390.265353, 385.411822, 365.307743, 383.715623, 349.72822, 138.893776, 91.413718, 88.316739, 93.425419, 105.406527, 89.589571, 86.302304, 272.626465, 307.17966, 337.463434, 281.092356, 279.199886, 306.292725, 316.569728, 227.391558, 55.923068, 44.952456, 51.19736, 53.55269, 51.558243, 45.767605, 42.046567, 39.563965, 265.502563, 215.856653, 242.076281, 231.645799, 244.051025, 242.815055, 275.07821, 275.07821, 0, 0, 0, 0, 0, 0, 0, 0, 170.323793, 208.373474, 230.702186, 207.303124, 207.031141, 216.519441, 209.39707, 194.040242, 0, 0, 0, 0, 0, 0, 0, 0, 135.361687, 206.205412, 219.142894, 216.055689, 206.04894, 208.74405, 223.424176, 220.788747, 62.552604, 0, 0, 0, 0, 0, 0, 14.732299, 211.572467, 199.751281, 185.313029, 208.148486, 196.409502, 195.367317, 191.607478, 194.640122, 81.261219, 0, 0, 0, 0, 0, 0, 34.758435, 34.758435, 224.880232, 228.967878, 201.218894, 230.38938, 217.632718, 220.296921, 192.772033, 0, 0, 0, 0, 0, 0, 0, 0, 108.154026, 235.846087, 222.978606, 197.172557, 225.031526, 212.861599, 182.86926, 41.131673, 0, 0, 0, 0, 0]
    
    
    # series = [1, 1, 1, 222.268281, 227.201073, 239.880936, 219.442175, 224.973491, 229.811911, 205.091031, 74.613006, 39.074674, 
    # 42.660607, 46.771171, 49.42565, 42.942462, 38.4036, 84.570203, 292.503893, 315.355352, 308.760204, 323.474822, 328.579046, 
    # 320.081387, 252.186329, 303.202204, 100.9927, 90.015986, 109.322007, 109.322007, 101.237994, 93.506046, 85.536672, 113.139345,
    #  363.6933, 336.765956, 336.765956, 360.514003, 360.514003, 386.828631, 333.582584, 311.141953, 137.747034, 126.938755, 153.468333,
    #   139.097676, 153.49441, 137.402691, 157.353422, 157.353422, 211.900311, 383.462928, 404.630162, 381.371638, 362.084191, 370.947801,
    #    377.015265, 399.585546, 123.408264, 113.998109, 112.856803, 106.889714, 118.416439, 120.229568, 112.551647, 149.311997, 360.410137,
    #     322.75247, 322.75247, 317.344449, 333.973616, 325.086149, 348.288513, 318.460927, 213.543097, 74.776173, 70.599969, 77.547022,
    #      71.492227, 60.858282, 72.131921, 228.631534, 287.955998, 252.905549, 255.666318, 293.224175, 293.224175, 254.890236, 256.31281, 
    #      184.259296, 26.746213, 20.837145, 25.642178, 20.129102, 20.129102, 28.922743, 21.160708, 152.93683, 200.704473, 226.081362, 
    #      196.194592, 200.608984, 202.260495, 173.108635, 179.423445, 17.397033, 17.397033, 0, 0, 0, 0, 0, 0, 98.800742, 192.672357,
    #       199.414745, 215.054186, 215.054186, 205.410493, 209.414565, 203.866979, 0, 0, 0, 0, 0, 0, 0, 0, 0, 176.747441, 223.099724, 
    #       194.794772, 207.239332, 210.721222, 186.568032, 202.327088, 63.881363, 0, 0, 0, 0, 0, 0, 0, 0, 203.325941, 203.325941]
         
    # series = [20, 21, 21, 20, 20, 21, 21, 20, 30, 29, 40, 41, 20, 21, 
    #         21, 20, 20, 21, 21, 20, 30, 29, 40, 39, 20, 21, 21, 20, 20, 21, 
    #         21, 20, 32, 29, 40, 41, 20, 21, 21, 19, 20, 23, 21, 20, 30, 29, 
    #         40, 42, 20, 21, 21, 20, 24, 21, 21, 20, 31, 30, 40, 41, 20, 21, 21]

    
    series = [0, 17.086175, 19.206876, 29.694794, 25.684078, 42.002275, 67.423028, 74.677777, 50.037768, 73.029972, 90.092406, 78.579672, 89.312967, 108.955076, 108.61438, 101.105229, 101.105229, 99.985247, 123.369904, 123.852829, 130.498571, 118.096587, 171.635449, 161.538299, 172.348394, 179.119746, 177.729565, 212.267127, 188.583961, 196.522142, 217.815906, 247.55892, 232.891279, 239.272192, 285.377823, 250.774545, 310.416354, 281.22997, 298.652927, 285.572187, 252.640196, 264.47059, 269.569383, 288.392294, 296.654045, 299.708194, 296.893402, 297.069059, 299.209823, 291.437301, 291.437301, 276.041278, 312.809196, 273.116851, 212.263057, 208.900272, 233.098872, 178.703479, 154.592896, 142.752804, 156.688798, 127.42866, 90.532862, 102.258999, 91.810945, 83.447442, 52.581879, 58.904436, 55.965987, 55.079667, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    # b = []
    # for i in series:
    #     b.extend([i, i])
    # series = b


    
    s_len = len(series)
    series = series +series
    # series = series +series 

    
    noise = np.random.normal(0,5,len(series))
    series = [sum(x) for x in zip(noise, series)]
    series = [int(i) for i in series]
    add = np.arange(len(series))
    add = [x*0.6 for x in add]
    series = [sum(x) for x in zip(add, series)]

    # Replace part of array
    # add = np.random.normal(1000,25,16)
    # series[240:256]=add
    # series[265:270]=add

    
    yrequest = []
    yhat = []
    Y = []

    


    #PARAMETERS
    params = {
        "window_future": 15, #HW
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
        "scaleup_count": 12, #FIX
        "scaledown_count": 12, #FIX
        "stable_range": 50

    }
    #scale buffer: only rescale when difference between prediction and usage is greater than x
    scale_d_b = [25, 50, 75, 100,150,200]
    scale_u_b = [25, 50, 75, 100,150,200]
    #scale stable: only scale when prediction has been stable for x steps
    scale_u_s = [3,4,5,10]
    scale_d_s = [3,4,5,10]
    #scale stable range: define what range is stable
    stable_range = [10, 25, 50, 75 , 100]


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
                            if rescale_counter < 15 and MSE < error: 

                                error = MSE
                                yrequest = yrequest_temp
                                best_params = params.copy()
                                
                        
    print(error)
    print(best_params)
    series = [0, 17.086175, 19.206876, 29.694794, 25.684078, 42.002275, 67.423028, 74.677777, 50.037768, 73.029972, 90.092406, 78.579672, 89.312967, 108.955076, 108.61438, 101.105229, 101.105229, 99.985247, 123.369904, 123.852829, 130.498571, 118.096587, 171.635449, 161.538299, 172.348394, 179.119746, 177.729565, 212.267127, 188.583961, 196.522142, 217.815906, 247.55892, 232.891279, 239.272192, 285.377823, 250.774545, 310.416354, 281.22997, 298.652927, 285.572187, 252.640196, 264.47059, 269.569383, 288.392294, 296.654045, 299.708194, 296.893402, 297.069059, 299.209823, 291.437301, 291.437301, 276.041278, 312.809196, 273.116851, 212.263057, 208.900272, 233.098872, 178.703479, 154.592896, 142.752804, 156.688798, 127.42866, 90.532862, 102.258999, 91.810945, 83.447442, 52.581879, 58.904436, 55.965987, 55.079667, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    s_len = len(series)
    series = series +series
    series = series +series 

    
    noise = np.random.normal(0,5,len(series))
    series = [sum(x) for x in zip(noise, series)]
    series = [int(i) for i in series]
    add = np.arange(len(series))
    add = [x*0.6 for x in add]
    series = [sum(x) for x in zip(add, series)]


    params = best_params
  
    
    yrequest = []
    rescale_counter = 0
    scaleup = 0
    downscale = 0
    CPU_request = 500
    i = 0
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
    
        yrequest.append(CPU_request)
        i += 1 

    series_X = range(len(series))
    ax1.plot(series_X, yrequest, '.-', linewidth=1,label='CPU request')
    ax1.plot(series_X, series, '.-', linewidth=1,label='CPU usage')
    #ax1.plot(series_X3, yhat, '.-', linewidth=1,label='Lib')

    
    
    ax1.set_ylim(bottom=0)
    txt=best_params
    plt.figtext(0.5, 0.01, txt, wrap=True, horizontalalignment='center', fontsize=12)
    ax1.legend(loc='lower center', bbox_to_anchor=(0.5, -0.25), fancybox=True, shadow=True, ncol=5)
    plt.show()

    

if __name__ == '__main__':
    main()