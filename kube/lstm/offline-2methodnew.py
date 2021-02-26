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
from pandas import read_csv
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

from keras.models import load_model

from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot

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
fig1 = plt.figure(1)
fig2 = plt.figure(2)
fig3 = plt.figure(3)
ax1 = fig1.add_subplot(1,1,1)
ax2 = fig2.add_subplot(1,1,1)
ax3 = fig3.add_subplot(1,1,1)

fig1.subplots_adjust(left=0.1, bottom=0.2, right=None, top=None, wspace=None, hspace=None)

# split a univariate sequence into samples
def split_sequence(sequence, n_steps_in, n_steps_out, ywindow):
    X, y = list(), list()

    for i in range(len(sequence)-ywindow-n_steps_in+1):
        # find the end of this pattern
        end_ix = i + n_steps_in

        # gather input and output parts of the pattern
        # print(sequence[end_ix:end_ix+ywindow])
        seq_x, seq_y = sequence[i:end_ix], [np.percentile(sequence[end_ix:end_ix+ywindow], params["lstm_target"]), np.percentile(sequence[end_ix:end_ix+ywindow], params["lstm_lower"]), np.percentile(sequence[end_ix:end_ix+ywindow], params["lstm_upper"])]
        X.append(seq_x)
        y.append(seq_y)

    # print(np.array(X), np.array(y))
    return np.array(X), np.array(y)

def trans_foward(arr):
    global scaler
    out_arr = scaler.transform(arr.reshape(-1, 1))
    return out_arr.flatten()

def trans_back(arr):
    global scaler
    out_arr = scaler.inverse_transform(arr.flatten().reshape(-1, 1))
    return out_arr.flatten()

def lstm_predict(input_data,model,n_steps_in,n_features):
    # demonstrate prediction
    
    x_input = np.array(trans_foward(input_data))
    x_input = x_input.reshape((1, n_steps_in, n_features))
    yhat = model.predict(x_input, verbose=0)
    return trans_back(yhat)

def train_test_split(data, n_test):
	return data[:n_test+1], data[-n_test:]

def calc_n(i):
    season = math.ceil((i+1)/params["season_len"])
    history_start_season = season - (params["history_len"]/params["season_len"])
    if history_start_season < 1:
        history_start_season = 1
    history_start = (history_start_season-1) * params["season_len"] 
    n = int(i - history_start)
    return n

def update_lstm(n_steps_in, n_steps_out, n_features,raw_seq, ywindow, model):
    global scaler
    raw_seq = np.array(raw_seq)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler = scaler.fit(raw_seq.reshape(-1, 1))
    
    dataset = trans_foward(raw_seq)
    # split into samples
    X, y = split_sequence(dataset, n_steps_in, n_steps_out, ywindow)
    # reshape from [samples, timesteps] into [samples, timesteps, features]
    
    X = X.reshape((X.shape[0], X.shape[1], n_features))

    model.fit(X, y, epochs=15, verbose=0)
    
    # model.fit(X[-144:,:,:], y[-144:], epochs=15, verbose=0)

    return model   


def create_lstm(n_steps_in, n_steps_out, n_features,raw_seq, ywindow):
    global scaler
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler = scaler.fit(raw_seq.reshape(-1, 1))
    #print("First 10 of raw_seq:", raw_seq[:20])
    dataset = trans_foward(raw_seq)
    # split into samples
    X, y = split_sequence(dataset, n_steps_in, n_steps_out, ywindow)
    # reshape from [samples, timesteps] into [samples, timesteps, features]
    X = X.reshape((X.shape[0], X.shape[1], n_features))
    # define model
    model = Sequential()
    
    # Multi-layer model 
    model.add(LSTM(50, return_sequences=True , input_shape=(n_steps_in, n_features)))
    # model.add(LSTM(50, return_sequences=True))
    model.add(LSTM(50))

    # Single layer model
    # model.add(LSTM(100, input_shape=(n_steps_in, n_features)))

    model.add(Dense(n_steps_out))
    model.compile(optimizer='adam', loss='mse')
    print(model.summary())
    # fit model
    # model.fit(X, y, epochs=15, verbose=1)

    
    # simple early stopping
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=3)
    mc = ModelCheckpoint('best_model.h5', monitor='val_loss', mode='min', verbose=1, save_best_only=True)
    # fit model
    history = model.fit(X, y, validation_split=0.3, epochs=30, verbose=0, callbacks=[es, mc])
    # load the saved model
    saved_model = load_model('best_model.h5')
    model = saved_model
    
    
    return model


def create_sin_noise(A, D, per, total_len):
    # Sine wave
    
    B = 2*np.pi/per
    x = np.arange(total_len)
    series = A*np.sin(B*x)+D
    #alpha = float(args.alpha)
    series = series * alpha

    noise = np.random.normal(0,int(std),len(series))*(1-alpha)
    series = [sum(x) for x in zip(noise, series)]
    series = [int(i) for i in series]

    series = np.array([1 if i <= 0 else i for i in series]).flatten()
    return series


params = {
    "window_future": 20, 
    "window_past": 1, 
    "lstm_target": 90, 
    "lstm_upper": 98, 
    "lstm_lower": 60, 
    "HW_target": 90, 
    "HW_upper": 98, 
    "HW_lower": 60, 
    "season_len": 144, 
    "history_len": 3*144, 
    "rescale_buffer": 120, 
    "rescale_cooldown": 20, 
}

slack_list_vpa = []
slack_list_hw = []
slack_list_lstm = []

perc_list_vpa = []
perc_list_hw = []
perc_list_lstm = []

above_list_vpa = []
above_list_hw = []
above_list_lstm = []

contname = "c_1"
def main():

    global params
    global alpha, std
    #c_1
    series_train = [480, 363, 480, 671, 643, 796, 978, 726, 241, 516, 1168, 1820, 1075, 1332, 168, 190, 208, 231, 897, 832, 535, 238, 242, 282, 221, 161, 148, 136, 124, 159, 152, 146, 139, 133, 126, 120, 310, 504, 697, 607, 262, 170, 286, 441, 596, 739, 878, 1016, 804, 520, 237, 1040, 1014, 988, 962, 1853, 2152, 2142, 2132, 2123, 2053, 2087, 2122, 2156, 2191, 2225, 2260, 2295, 2329, 2364, 2398, 2433, 2467, 2502, 2536, 2571, 2605, 2640, 2675, 2709, 2339, 1889, 1438, 987, 752, 980, 731, 1714, 2267, 2238, 2209, 2180, 2151, 2122, 2093, 2064, 2035, 2006, 1977, 1948, 1919, 1890, 1861, 1832, 1803, 1774, 1745, 1716, 1687, 1658, 1630, 1601, 1572, 1543, 1514, 1485, 1456, 1427, 1398, 1369, 1340, 1311, 1282, 1253, 1486, 1165, 736, 702, 403, 859, 1000, 846, 825, 953, 456, 287, 493, 715, 697, 527, 274, 763, 626, 489, 333, 235, 369, 386, 412, 494, 576, 658, 388, 620, 360, 781, 546, 472, 405, 192, 340, 204, 340, 520, 696, 780, 863, 947, 360, 360, 363, 402, 444, 513, 581, 316, 372, 352, 223, 466, 482, 296, 480, 489, 549, 128, 186, 1134, 1664, 1491, 1318, 1145, 971, 731, 966, 989, 488, 856, 908, 1349, 1325, 1302, 1314, 1974, 2268, 2533, 1855, 1506, 1945, 2384, 1148, 2066, 1835, 1604, 1177, 1311, 1023, 990, 909, 1160, 1411, 1136, 651, 696, 741, 786, 831, 1066, 806, 1045, 1184, 1901, 1338, 793, 1472, 1910, 2064, 1987, 857, 1721, 1493, 2005, 2028, 1216, 957, 1270, 1581, 1704, 1319, 922, 1558, 796, 780, 764, 760, 760, 765, 869, 973, 853, 679, 914, 1148, 1360, 1360, 749, 477, 432, 310, 553, 1269, 1941, 2081, 1508, 653, 1081, 932, 626, 560, 560, 840, 1020, 702, 1288, 506, 731, 825, 274, 537, 654, 770, 634, 497, 361, 293, 326, 359, 392, 425, 458, 492, 525, 558, 370, 399, 505, 611, 168, 291, 320, 401, 536, 633, 604, 650, 708, 743, 620, 497, 374, 256, 120, 680, 1028, 492, 203, 239, 367, 527, 543, 559, 387, 874, 814, 773, 563, 458, 805, 1153, 1500, 1520, 1650, 1576, 970, 2744, 1974, 1348, 1915, 2046, 2238, 2326, 2414, 2309, 2056, 2112, 2234, 2367, 2500, 2806, 2396, 1976, 1686, 1930, 1680, 1660, 2249, 1295, 1351, 2018, 1678, 1368, 1497, 1627, 1668, 1700, 1891, 1672, 1656, 1640, 1352, 1350, 2528, 1697, 2120, 1464, 1392, 1534, 1326, 1099, 2163, 1781, 1820, 2012, 2204, 1682, 1445, 2144, 2217, 1415, 1258, 1466, 1360, 1436, 1738, 1401, 1460, 1453, 1164, 1340, 1516, 1545, 1231, 1571, 1065, 1108, 1048, 1203, 1304, 1040, 1568, 1276, 933, 728, 621, 514, 407, 300, 249, 716, 648, 587, 525, 443, 409, 522, 1240, 380, 532, 321, 746, 784, 601, 350, 126, 201, 276, 351, 426, 805, 276, 184, 330, 440, 358, 179, 544, 194, 216, 272, 338, 725, 194, 320, 745, 697, 989, 1101, 961, 149, 442, 330, 245, 205, 234, 274, 315, 202, 560, 1781, 1204, 1246, 1077, 1097, 1557, 770, 696, 1256, 1631, 1610, 1739, 1617, 1496, 1374, 1252, 1894, 1350, 1320, 1713, 1445, 1403, 1851, 1493, 1135, 777, 760, 854, 723, 1644, 1188, 1914, 2035, 1713, 1030, 757, 720, 682, 1058, 1781, 1446, 656, 613, 844, 693, 795, 896, 997, 1239, 1157, 1058, 876, 732, 588, 444, 855, 1256, 1889, 1948, 1229, 513, 506, 626, 592, 538, 535, 559, 731, 451, 1137, 1528, 1642, 489, 1052, 1614, 1538, 972, 848, 912, 754, 337, 422, 944, 377, 429, 481, 480, 334, 280, 288, 553, 818, 1241, 710, 684, 359, 344, 330, 252, 501, 225, 199, 173, 405, 426, 261, 145, 116, 210, 556, 356, 154, 137, 120, 429, 537, 417, 179, 188, 418, 167, 304, 441, 577, 714, 194, 306, 414, 522, 160, 293, 103, 160, 160, 160, 183, 415, 448, 318, 189, 266, 579, 398, 812, 845, 872, 373, 402, 432, 516, 721, 1056, 1047, 949, 936, 923, 1187, 871, 683, 786, 889, 1996, 1250, 1372, 1325, 607, 811, 1220, 1168, 845, 1493, 604, 638, 486, 812, 1300, 1200, 761, 822, 856, 625, 559, 510, 544, 719, 894, 1057, 1161, 1266, 1332, 1052, 984, 1718, 1879, 1072, 1135, 1235, 1518, 921, 886, 1003, 1120, 1164, 1112, 1115, 1878, 662, 683, 981, 586, 679, 819, 864, 903, 829, 572, 815, 825, 782, 1046, 459, 500, 540, 640, 250, 497, 520, 785, 865, 648, 464, 675, 258, 299, 568, 1260, 716, 532, 392, 1451, 1442, 1062, 712, 636, 364, 405, 434, 463, 526, 637, 748, 747, 647, 548, 449, 350, 250, 177, 382, 562, 579, 597, 133, 160, 160, 220, 701, 592, 482, 373, 268, 194, 119, 176, 249, 154, 144, 134, 125, 260, 521, 678, 625, 572, 234, 426, 521, 419, 496, 1397, 1327, 825, 958, 800, 1440, 1606, 1772, 1864, 1492, 1557, 1621, 1685, 1750, 1413, 1013, 648, 1696, 1947, 2098, 2422, 2825, 3161, 2901, 2641, 2381, 2168, 2935, 2493, 2099, 2390, 2681, 2972, 1344, 2316, 2082, 2547, 2023, 1499, 923, 943, 964, 984, 957, 798, 639, 738, 1053, 1450, 1825, 1941, 1832, 1723, 1614, 2010, 1618, 1359, 1695, 2184, 2253, 798, 1146, 1680, 2240, 2411, 2508, 2605, 2703, 2323, 1831, 1339, 836, 1926, 1561, 1410, 1523, 1636, 1749, 1862, 1975, 2040, 2040, 2041, 2053, 2064, 2076, 2194, 2356, 2354, 2303, 2252, 2201, 2148, 2088, 2182, 2011, 2036, 2123, 2210, 2019, 2076, 2001, 1918, 1834, 1819, 1709, 1712, 1910, 2148, 2314, 2158, 2002, 1967, 2225, 1849, 1912, 1975, 2038, 2059, 2078, 1958, 1828, 1784, 1917, 2007, 1661, 2079, 2061, 2044, 2040, 1589, 1968, 2148, 2259, 2197, 2153, 2006, 1859, 2110, 2087, 2456, 2993, 3077, 2878, 2681, 2489, 2072, 1714, 1439, 1164, 889, 1756, 1585, 2762, 2034, 1092, 1041, 1191, 1340, 1380, 1346, 1313, 1280, 1126, 1586, 1698, 1221, 880, 1082, 1284, 1486, 1688, 1899, 893, 943, 920, 657, 679, 702, 774, 1046, 1235, 1186, 1137, 1482, 2714, 2635, 1788, 2016, 2076, 1203, 898, 622, 2934, 2504, 2352, 2285, 1532, 2057, 3120, 3120, 3036, 2919, 1131, 1156, 1430, 2423, 1178, 1779, 2234, 1066, 1040, 2150, 1458, 803, 1675, 2316, 967, 678, 638, 571, 504, 448, 695, 771, 658, 470, 444, 418, 385, 332, 279, 263, 506, 215, 224, 199, 175, 261, 469, 1036, 1271, 1174, 1076, 979, 882, 784, 641, 324, 368, 358, 218, 438, 392, 345, 299, 252, 206, 201, 257, 418, 200, 204, 232, 220, 198, 179, 160, 160, 160, 147, 81, 164, 194, 185, 177, 168, 164, 226, 280, 303, 243, 253, 268, 361, 445, 957, 2107, 1823, 1539, 1120, 465, 575, 851, 979, 1163, 1571, 1145, 812, 958, 1293, 1429, 1371, 1354, 1192, 1070, 1349, 1624, 1362, 870, 1018, 878, 1718, 1670, 1542, 1263, 775, 1472, 1067, 960, 942, 907, 872, 838, 803, 768, 733, 698, 892, 1698, 1554, 1703, 2062, 2422, 2213, 1501, 1295, 1162, 1052, 817, 1266, 1280, 1636, 1536, 772, 1034, 1271, 785, 690, 762, 834, 647, 802, 580, 760, 760, 1152, 2272, 848, 711, 788, 866, 817, 532, 630, 1984, 1925, 852, 2046, 724, 791, 228, 588, 655, 1842, 1212, 202, 240, 570, 337, 1389, 1402, 960, 517]
    #c_10235
    #series_train = [920, 859, 799, 739, 679, 607, 769, 584, 438, 430, 423, 416, 408, 401, 393, 386, 378, 371, 363, 356, 349, 341, 334, 326, 311, 341, 336, 346, 341, 320, 320, 240, 243, 317, 440, 245, 383, 440, 440, 440, 440, 407, 293, 372, 433, 391, 400, 420, 441, 503, 566, 629, 689, 739, 827, 947, 1085, 1000, 1077, 1056, 1060, 1192, 1398, 1560, 1722, 1707, 1639, 1571, 1490, 1387, 1336, 1286, 1235, 1190, 1160, 1129, 974, 901, 1016, 1014, 1081, 851, 928, 1029, 1136, 1296, 1335, 1204, 1047, 994, 986, 977, 969, 961, 953, 945, 936, 928, 920, 912, 904, 895, 887, 879, 871, 863, 854, 846, 838, 830, 822, 813, 805, 797, 789, 781, 772, 764, 756, 748, 740, 731, 723, 885, 880, 858, 968, 791, 691, 925, 736, 825, 965, 838, 696, 857, 778, 864, 799, 746, 878, 787, 695, 604, 600, 548, 480, 480, 480, 652, 520, 520, 477, 430, 450, 560, 447, 448, 360, 360, 360, 360, 360, 360, 360, 360, 306, 249, 309, 369, 440, 240, 240, 244, 320, 240, 240, 373, 382, 321, 460, 405, 349, 293, 385, 280, 305, 281, 394, 380, 386, 441, 495, 550, 605, 680, 880, 670, 749, 828, 906, 1181, 1046, 960, 1148, 1376, 1323, 1240, 1159, 1280, 1256, 1218, 1180, 1270, 1136, 1117, 1240, 1240, 1240, 1240, 1226, 1170, 1099, 1090, 1138, 1137, 1016, 1049, 1080, 1153, 1041, 1264, 1446, 1385, 1160, 1018, 1240, 1132, 1158, 1207, 1259, 1310, 1356, 1244, 1133, 951, 990, 1094, 995, 1272, 1054, 1005, 933, 857, 861, 951, 800, 754, 789, 824, 685, 713, 742, 1053, 1045, 918, 838, 913, 935, 846, 879, 911, 746, 951, 926, 889, 849, 809, 948, 838, 790, 767, 833, 814, 747, 800, 800, 903, 901, 804, 707, 640, 653, 666, 679, 664, 648, 665, 557, 577, 678, 484, 511, 640, 632, 609, 557, 503, 474, 446, 417, 389, 360, 438, 355, 280, 308, 400, 400, 351, 315, 304, 293, 282, 612, 332, 396, 436, 409, 356, 289, 319, 349, 505, 456, 523, 582, 616, 560, 664, 700, 724, 749, 774, 799, 1120, 1143, 1458, 1218, 1113, 1156, 1306, 1285, 1268, 1253, 1238, 1223, 1208, 1158, 1210, 1225, 1044, 1510, 1411, 1313, 1215, 1116, 1045, 1072, 1099, 1345, 1169, 887, 843, 1033, 1160, 1160, 1160, 1225, 1270, 1053, 1129, 1232, 1326, 1214, 1058, 1336, 1221, 1070, 1000, 1000, 1000, 1000, 1021, 1068, 1116, 1141, 915, 817, 800, 824, 686, 819, 733, 773, 813, 853, 893, 933, 778, 924, 812, 749, 983, 896, 748, 1011, 924, 1118, 1022, 925, 829, 732, 802, 520, 754, 628, 751, 680, 680, 698, 753, 717, 692, 592, 683, 694, 562, 683, 680, 441, 513, 381, 373, 387, 395, 365, 360, 360, 360, 360, 360, 360, 360, 360, 360, 320, 314, 307, 300, 293, 286, 280, 315, 360, 360, 380, 317, 320, 297, 349, 538, 284, 330, 387, 337, 367, 402, 484, 505, 520, 701, 730, 516, 711, 904, 820, 999, 825, 861, 1033, 1040, 1059, 1251, 1344, 1420, 1496, 1479, 1372, 1280, 1280, 1465, 1136, 1160, 1116, 1068, 1323, 1036, 1110, 1185, 1200, 1200, 1200, 1200, 1200, 1200, 1200, 1200, 1174, 1122, 1074, 1048, 1022, 1038, 1131, 1148, 1170, 1214, 1147, 1177, 1093, 1134, 1147, 1162, 1170, 1135, 1100, 1053, 989, 925, 862, 798, 990, 660, 971, 948, 800, 652, 701, 538, 836, 764, 852, 818, 945, 918, 891, 720, 963, 916, 870, 823, 776, 730, 837, 730, 676, 892, 852, 739, 626, 513, 602, 626, 633, 620, 607, 639, 619, 600, 580, 560, 540, 520, 500, 480, 535, 560, 435, 401, 497, 525, 552, 360, 360, 319, 297, 384, 310, 288, 320, 339, 320, 367, 437, 377, 338, 360, 291, 378, 490, 477, 403, 289, 257, 293, 325, 393, 389, 456, 533, 517, 454, 440, 440, 456, 501, 547, 592, 637, 724, 814, 866, 904, 960, 1061, 1228, 852, 1257, 980, 1223, 1198, 1186, 1441, 1295, 1036, 1357, 794, 766, 1026, 1072, 1018, 1196, 1030, 1024, 917, 840, 864, 948, 1115, 1005, 817, 1024, 1254, 1080, 1080, 1116, 1025, 1008, 1067, 1019, 928, 987, 1133, 1009, 955, 949, 943, 937, 931, 924, 915, 894, 901, 875, 814, 752, 691, 810, 810, 836, 761, 828, 765, 760, 780, 813, 846, 880, 913, 791, 870, 842, 830, 800, 800, 640, 640, 640, 901, 807, 712, 618, 655, 693, 673, 683, 774, 829, 786, 669, 600, 600, 740, 714, 653, 757, 597, 542, 488, 483, 458, 389, 397, 463, 531, 424, 389, 338, 281, 536, 549, 417, 364, 320, 338, 283, 308, 305, 218, 357, 480, 402, 341, 250, 300, 288, 399, 310, 297, 284, 271, 349, 421, 450, 420, 381, 476, 584, 600, 600, 600, 729, 823, 900, 977, 941, 903, 1039, 1175, 1182, 1286, 1238, 1191, 1179, 1143, 1190, 1320, 1270, 1446, 1256, 1080, 988, 935, 1004, 1041, 1077, 1113, 1161, 984, 1104, 1200, 1217, 1234, 1089, 1118, 1219, 1319, 1376, 1260, 1143, 1209, 1321, 1434, 1535, 1600, 1456, 1271, 1257, 1244, 1043, 1055, 1067, 1079, 1080, 812, 890, 967, 1029, 920, 1099, 788, 1045, 900, 858, 1188, 900, 843, 760, 721, 839, 957, 1074, 1034, 1000, 974, 844, 902, 960, 1018, 760, 798, 932, 1026, 957, 704, 772, 797, 760, 760, 728, 690, 653, 680, 680, 680, 651, 640, 645, 670, 614, 560, 469, 445, 513, 537, 545, 484, 422, 426, 468, 446, 542, 349, 421, 360, 333, 389, 324, 280, 280, 290, 328, 273, 265, 330, 360, 360, 360, 360, 360, 358, 360, 280, 445, 400, 400, 394, 280, 309, 338, 376, 457, 491, 516, 640, 916, 828, 880, 880, 880, 893, 984, 980, 1260, 998, 1257, 1425, 1275, 1125, 990, 1077, 1080, 1096, 1113, 1164, 1005, 1028, 1080, 1126, 988, 1075, 1188, 1045, 1137, 1159, 1134, 1137, 1180, 1241, 962, 1095, 1228, 1071, 1054, 1047, 1068, 1094, 1214, 1033, 904, 1147, 1158, 1116, 1003, 1024, 1037, 1088, 1138, 1189, 1160, 1160, 1093, 925, 1124, 1175, 1136, 1097, 947, 1000, 895, 904, 933, 848, 850, 961, 1003, 1045, 1087, 1000, 786, 984, 920, 880, 1021, 1126, 840, 840, 840, 840, 935, 880, 824, 769, 725, 684, 720, 720, 720, 699, 686, 772, 686, 624, 579, 446, 422, 405, 400, 400, 443, 528, 560, 507, 520, 398, 354, 294, 509, 422, 360, 298, 331, 405, 320, 320, 351, 373, 318, 272, 245, 294, 358, 320, 280, 280, 280, 288, 342, 396, 450, 504, 520, 520, 520, 400, 477, 631, 468, 620, 640, 713, 892, 1066, 1081, 1100, 1121, 1166, 1196, 1225, 1157, 1228, 1357, 1301, 1191, 1081, 1360, 1110, 1172, 1289, 1077, 864, 850, 862, 874, 998, 1029, 1080, 1113, 1147, 1486, 1119, 1176, 1232, 1286, 1317, 1296, 1243, 1083, 1152, 1118, 1084, 1080, 1080, 1247, 1301, 1200, 1131, 1308, 1240, 1276, 1052, 1106, 1279, 1249, 1218, 1188, 1158, 1128, 966, 949, 939, 928, 901, 890, 801, 815, 770, 924, 992, 1034, 856, 818, 868, 1000, 1060, 1057, 1086, 1114, 920, 869, 877, 842, 806, 800, 800]
    series_train = series_train[144:]
    series = series_train
    series_train = np.concatenate((series_train, series_train, series_train, series_train))
    
    #c_10235
    # avg HW: 275.7849532977749
    # avg LSTM: 270.7894836767715
    # avg VPA: 468.390756302521
    # ---% TIME ABOVE REQUESTED---
    # perc above vpa_target:  [5.319999999999999]
    # perc above hw:  [4.06]
    # perc above lstm:  [3.64]
    
    # series = [0, 0, 61.743225, 65.812928, 64.688258, 61.283377, 61.283377, 67.654982, 68.506236, 68.069168, 63.33223, 67.350238, 67.350238, 59.135644, 59.135644, 65.063693, 65.367389, 64.644802, 64.644802, 63.671429, 67.479587, 60.369221, 63.746355, 68.426908, 68.426908, 67.848495, 68.379601, 65.033326, 66.127736, 66.072899, 64.005658, 65.626443, 65.626443, 66.947988, 64.115421, 65.943284, 68.052725, 66.706593, 64.079707, 66.359751, 66.652653, 70.252121, 62.923355, 67.126152, 63.567186, 63.522037, 65.607079, 65.990171, 62.548382, 61.127716, 62.720609, 64.266515, 65.87071, 65.87071, 65.245995, 57.512539, 59.729175, 64.033217, 63.277707, 63.300236, 65.711968, 63.283551, 64.754742, 64.754742, 68.782502, 63.018085, 63.018085, 63.563587, 71.889895, 64.555834, 62.135457, 61.660707, 62.18573, 62.18573, 67.263784, 71.418236, 137.838296, 124.973905, 144.046814, 188.035742, 186.621325, 198.083011, 247.179403, 244.414532, 290.002148, 290.002148, 295.501502, 340.193935, 337.603942, 337.603942, 337.75654, 342.075036, 340.059003, 357.540656, 333.370868, 378.765409, 384.203911, 376.488329, 417.234252, 419.647498, 399.808957, 420.418237, 417.371609, 415.687412, 395.31032, 409.665596, 414.328398, 367.193016, 380.588229, 355.168267, 351.698839, 331.657607, 338.393024, 338.393024, 323.316553, 323.316553, 335.339155, 303.564811, 304.098715, 250.799748, 250.799748, 252.688684, 228.040479, 195.5729, 181.70091, 135.338763, 135.338763, 107.944042, 107.944042, 66.244164, 61.989939, 66.483219, 65.621958, 64.934725, 66.901524, 66.69103, 64.781212, 67.526626, 67.774357, 65.041874, 70.976161, 69.392448, 67.182445, 68.313569, 65.735736, 69.748069, 69.879586, 68.653724, 63.957029, 63.507327, 68.12431, 66.650802, 65.715767, 60.000721, 67.136805, 63.062296, 64.018353, 62.040963, 62.040963, 63.679135, 62.154706, 65.181957, 64.068167, 62.955775, 63.776876, 64.813811, 67.133909, 63.064697, 63.378305, 64.617083, 65.588123, 63.81287, 67.709429, 65.932648, 69.467776, 68.323161, 66.588076, 64.669751, 63.069118, 67.945729, 67.22985, 66.832906, 66.23627, 65.847554, 65.495303, 65.495303, 66.514129, 67.329368, 65.083671, 61.357294, 62.94477, 62.212584, 68.336662, 67.07261, 64.864739, 64.958125, 69.507114, 66.610447, 68.525375, 65.489864, 68.253549, 65.888995, 70.468058, 75.914542, 68.394364, 69.441889, 65.384635, 67.455503, 67.455503, 68.250114, 65.357909, 65.483581, 65.483581, 71.977791, 72.614541, 72.614541, 67.588842, 63.59081, 68.024119, 63.750568, 120.006415, 141.780724, 157.363302, 194.491844, 191.004811, 214.548369, 258.842478, 247.114835, 312.71089, 301.050892, 290.10112, 329.702207, 324.740036, 327.989362, 341.644953, 345.155229, 339.429658, 349.754034, 343.547861, 381.285517, 370.047963, 370.047963, 404.025968, 406.407655, 395.729498, 410.945677, 408.927901, 388.462345, 411.756913, 413.74045, 379.053994, 377.120611, 374.794915, 342.678069, 350.37731, 351.175823, 330.527249, 339.084112, 314.821851, 345.557808, 335.729661, 306.96319, 294.292478, 279.71473, 263.332449, 263.332449, 252.101558, 197.453348, 193.687384, 151.860015, 136.625863, 138.461229, 68.300933, 65.568534, 59.355894, 66.259219, 70.144654, 69.622521, 66.224944, 66.347775, 67.625393, 71.684717, 69.288331, 67.552996, 66.877703, 65.358259, 64.09007, 65.972771, 46.733962, 46.733962, 59.836314, 67.122512, 67.122512, 69.136134, 69.565368, 66.371765, 68.182048, 66.810451, 64.643932, 60.62675, 68.080732, 66.749273, 64.798237, 64.798237, 64.878494, 63.181709, 65.372129, 64.0131, 65.70622, 69.175132, 66.861682, 64.719033, 67.670241, 64.220562, 68.25315, 67.834423, 68.560337, 69.022853, 69.313214, 65.118357, 67.406357, 63.376912, 65.045172, 64.989495, 63.718965, 63.345824, 65.449804, 64.155342, 62.886053, 66.267959, 65.485001, 65.855195, 69.179225, 66.506739, 66.506739, 65.672874, 65.700215, 62.53296, 68.086563, 66.499444, 62.409591, 62.711345, 62.659603, 62.659603, 60.395247, 66.284063, 65.326346, 67.663318, 66.458327, 67.466311, 70.133281, 65.957265, 66.595222, 67.34767, 63.437278, 63.25499, 67.629983, 66.231469, 67.961705, 68.507882, 67.745954, 67.745954, 67.248327, 62.504716, 94.801841, 137.00814, 145.935319, 190.619658, 194.037242, 233.892233, 233.892233, 254.712385, 261.488186, 298.071202, 282.161501, 329.107831, 317.119975, 312.571976, 341.597658, 340.412251, 341.760491, 355.978433, 354.902114, 363.21097, 381.047892, 371.724801, 414.718742, 411.074446, 399.373427, 409.639409, 410.727897, 393.188453, 410.719083, 415.898426, 392.934029, 377.301244, 379.155277, 379.155277, 349.539475, 335.751355, 335.751355, 340.206766, 344.070271, 315.490784, 315.490784, 292.290156, 304.459899, 302.161018, 255.155852, 255.155852, 203.190348, 187.090546, 193.63851, 146.449029, 133.040927, 133.040927, 63.658729, 65.872208, 65.872208, 63.785919, 65.557305, 66.257614, 66.257614, 64.719509, 69.977905, 63.764268, 72.086588, 72.086588, 79.028644, 72.453438, 72.453438, 71.439419, 65.766206, 67.981376, 69.559621, 66.739438, 72.002492, 64.164944, 64.655924, 64.272103, 64.272103, 62.007494, 62.007494, 66.338223, 71.645633, 71.645633, 63.56426, 64.926313, 66.310069, 64.223916, 70.085048, 68.419701, 71.077381, 71.36494, 64.996268, 70.869812, 72.071281, 70.278834, 66.573504, 64.79114, 67.027187, 62.453225, 62.453225, 64.880695, 62.802072, 65.361954, 63.988112, 68.05674, 68.05674, 67.980978, 69.14486, 67.490425, 71.864219, 63.439282, 64.259946, 66.794305, 65.083107, 63.974744, 64.744144, 66.039328, 63.075114, 64.599929, 69.433168, 67.252682, 69.62072, 68.596196, 65.898732, 65.898732, 64.297847, 66.054371, 67.160351, 66.757853, 66.550352, 65.17163, 65.384049, 71.176669, 68.427492, 68.479803, 67.810531, 67.852539, 69.594607, 69.594607, 70.910444, 68.897539, 70.113685, 70.113685, 62.509112, 65.087484, 133.947896, 133.947896, 132.924342, 165.104443, 188.169863, 214.01004, 257.470853, 240.649592, 308.856074, 309.477951, 290.347634, 322.716367, 333.424267, 333.424267, 345.628135, 340.405617, 342.477336, 352.506857, 345.058327, 383.199048, 384.811998, 384.811998, 417.381313, 417.381313, 378.148455, 411.174043, 409.55963, 409.55963, 390.938918, 407.548575, 392.441869, 382.398889, 382.757602, 335.682254, 335.682254, 355.582738, 349.104937, 339.425138, 311.241604, 325.765666, 336.398238, 336.398238, 277.215851, 297.545856, 254.817334, 254.882559, 244.950759, 187.152664, 182.643752, 135.005404, 135.005404, 137.099465, 93.870301, 67.082988, 65.723279, 64.320887, 69.562654, 74.428905, 65.510198, 65.510198, 62.158134, 65.298171, 67.352314, 65.257533, 64.268172, 64.268172, 64.564241, 62.156164, 62.156164, 66.038028, 64.117921, 64.117921, 65.074717, 68.808897, 61.544367, 61.544367, 63.719097, 66.237215, 67.173829, 61.888583, 66.855439, 69.535783, 71.224133, 67.677779, 65.835607, 67.041732, 63.31931, 62.19923, 62.33675, 63.082979, 67.750766, 67.750766, 65.903316, 65.559874, 66.629098, 66.887576, 63.563459, 67.011674, 61.641178, 65.090452, 67.282378, 67.282378, 62.226595, 71.180984, 63.285314, 63.569488, 62.547156, 67.443055, 62.172521, 62.172521, 68.00451, 66.581996, 66.23648, 66.102292, 67.380522, 67.380522, 68.871685, 68.871685, 64.193322, 66.494357, 60.046868, 60.046868, 64.822877, 64.009062, 67.008353, 65.557842, 65.797809, 65.164547, 61.930419, 68.302495, 68.302495, 62.909795, 70.484541, 64.651806, 64.789164, 69.860725, 65.664353, 66.606772, 67.59384, 68.336629, 69.146114, 64.815284, 67.871863, 69.952916, 121.959888, 136.68046, 139.548716, 197.362649, 197.362649, 182.25515, 254.721091, 246.582766, 246.582766, 295.386888, 295.060452, 327.11073, 327.11073, 316.816171, 349.175202, 328.152328, 352.463196, 349.979274, 342.156759, 382.986456, 383.414558, 375.817267, 409.974599, 411.372933, 410.578232, 414.90799, 418.508568, 398.545694, 411.562525, 411.407412, 411.788265, 364.112456, 364.112456, 339.263475, 351.129407, 323.364911, 337.980744, 333.326596, 333.326596, 335.167825, 330.241032, 280.122562, 287.321545, 297.07834, 279.535228, 254.74456, 254.374602, 179.04024, 188.421524, 189.60678, 189.60678, 134.465954, 105.120722, 64.622324, 64.037213, 60.449515, 67.349291, 66.888078, 66.888078, 65.276643, 65.22202, 66.109513, 63.836765, 64.270034, 65.032251, 71.107685, 71.107685, 63.12264]
    # series = np.array(series)

    # Two patterns
    # series1 = create_sin_noise(A=300, D=200, per=params["season_len"], total_len=4*params["season_len"])
    # series2 = create_sin_noise(A=700, D=400, per=params["season_len"], total_len=3*params["season_len"])
    # series = np.concatenate((series1,series2), axis=0)
    # series = np.concatenate((series,series1), axis=0)

    scaling_start_index = params["season_len"]*2

    hw_CPU_request = 2000
    lstm_CPU_request = 2000

    lstm_requests = [lstm_CPU_request] * scaling_start_index
    lstm_targets = []
    lstm_uppers = []
    lstm_lowers = []

    hw_requests = [hw_CPU_request] * scaling_start_index
    hw_targets = []
    hw_uppers = []
    hw_lowers = []
    

    i = scaling_start_index 

    lstm_cooldown = 0
    hw_cooldown = 0

    model = None
    hw_model = None

    steps_in, steps_out, n_features, ywindow = 144, 3, 1, params["window_future"]

    # Start autoscaling only after we have gathered 2 seasons of data
    while i <= len(series):
        if i % 144 == 0:

            print(i)
        # Series up until now
        series_part = series[:i]
        n = calc_n(i)
        # Update/create HW model 
        if i % 1 == 0 or hw_model is None:
        # if hw_model is None:
            hw_model = ExponentialSmoothing(series_part[-n:], trend="add", seasonal="add", seasonal_periods=params["season_len"])
            model_fit = hw_model.fit()
        # HW prediction
        hw_window = model_fit.predict(start=n-params["window_past"],end=n+params["window_future"])
        hw_target = np.percentile(hw_window, params["HW_target"])
        hw_lower = np.percentile(hw_window, params["HW_lower"])
        hw_upper = np.percentile(hw_window, params["HW_upper"])

        
        if model is None: 
            model = create_lstm(steps_in, steps_out,n_features, np.array(series_train), ywindow)
            
            
        # LSTM prediction
        input_data = np.array(series_part[-steps_in:])
        output_data = lstm_predict(input_data, model,steps_in, n_features)

        lstm_target = output_data[0] # Target percentile value
        lstm_lower = output_data[1] # Lower bound value 
        lstm_upper = output_data[2] # upper bound value 

        # Keep CPU values above 0 
        if lstm_target < 0:
            lstm_target = 0
        if lstm_lower < 0:
            lstm_lower = 0
        if lstm_upper < 0:
            lstm_upper = 0

        if hw_target < 0:
            hw_target = 0
        if hw_lower < 0:
            hw_lower = 0
        if hw_upper < 0:
            hw_upper = 0

        lstm_targets.append(lstm_target)
        lstm_uppers.append(lstm_upper)
        lstm_lowers.append(lstm_lower)

        hw_targets.append(hw_target)
        hw_uppers.append(hw_upper)
        hw_lowers.append(hw_lower)


        # HW scaling 
        hw_CPU_request_unbuffered = hw_CPU_request - params["rescale_buffer"]
        # If no cool-down
        if (hw_cooldown == 0):
            # If request change greater than 50
            if (abs(hw_CPU_request - (hw_target + params["rescale_buffer"])) > 50):
                # If above upper
                if hw_CPU_request_unbuffered > hw_upper:
                    hw_CPU_request = hw_target + params["rescale_buffer"]
                    hw_cooldown = params["rescale_cooldown"]
                # elseIf under lower
                elif hw_CPU_request_unbuffered < hw_lower: 
                    hw_CPU_request = hw_target + params["rescale_buffer"]
                    hw_cooldown = params["rescale_cooldown"]

        # Reduce cooldown 
        if hw_cooldown > 0:
            hw_cooldown -= 1

        hw_requests.append(hw_CPU_request)

        # LSTM scaling
        lstm_CPU_request_unbuffered = lstm_CPU_request - params["rescale_buffer"]
        
        # If no cool-down
        if (lstm_cooldown == 0):
            # If request change greater than 50
            if (abs(lstm_CPU_request - (lstm_target + params["rescale_buffer"])) > 50):
                # If above upper
                if lstm_CPU_request_unbuffered > lstm_upper:
                    lstm_CPU_request = lstm_target + params["rescale_buffer"]
                    lstm_cooldown = params["rescale_cooldown"]
                # elseIf under lower
                elif lstm_CPU_request_unbuffered < lstm_lower: 
                    lstm_CPU_request = lstm_target + params["rescale_buffer"]
                    lstm_cooldown = params["rescale_cooldown"]

        # Reduce cooldown 
        if lstm_cooldown > 0:
            lstm_cooldown -= 1

        lstm_requests.append(lstm_CPU_request)


        i += 1

    
    X_targets = range(len(lstm_targets))
    X_targets = [x+scaling_start_index for x in X_targets]
    X_requests = range(len(lstm_requests))
    series_X = range(len(series))
    
    # PLOT 1 predictions, order: CPU, VPA, HW, LSTM, HW b, LSTM b
    
    ax1.plot(series_X, series, 'y-', linewidth=1,label='CPU usage')

    #Plot estimate of VPA target
    target = np.percentile(series, 90) + 50
    vpa_target = [target] * len(series_X)

    ax1.plot(series_X, vpa_target, 'g--', linewidth=1,label='VPA target')
    
    ax1.plot(X_targets, hw_targets, 'r-', linewidth=2,label='HW target')

    ax1.plot(X_targets, lstm_targets, 'b--', linewidth=2,label='LSTM target')

    ax1.fill_between(X_targets, hw_lowers, hw_uppers, facecolor='red', alpha=0.3, label="HW bounds")
    ax1.fill_between(X_targets, lstm_lowers, lstm_uppers, facecolor='blue', alpha=0.3, label="LSTM bounds")    




    t = ("Container "+ contname)
    fig1.suptitle(t, fontsize=23)
    
    ax1.tick_params(axis="x", labelsize=20) 
    ax1.tick_params(axis="y", labelsize=20) 

    leg1 = ax1.legend(loc='lower center', bbox_to_anchor=(0.5, -0.30), fancybox=True, shadow=True, ncol=6, fontsize=15)
    leg1_lines = leg1.get_lines()
    plt.setp(leg1_lines, linewidth=5)

    ax1.set_xlabel('Observations', fontsize=20)
    ax1.set_ylabel('CPU (millicores)', fontsize=20)

    # PLOT 2 Slack
    hw_slack = np.subtract(hw_requests[:-1],series)
    lstm_slack = np.subtract(lstm_requests[:-1],series)
    vpa_slack = np.subtract(vpa_target,series)
    #lstm_slack = [0 if i < 0 else i for i in lstm_slack]
    #vpa_slack = [0 if i < 0 else i for i in vpa_slack]

    ax2.plot(series_X, vpa_slack, 'y--', linewidth=1, label='VPA slack')
    ax2.plot(series_X, hw_slack, 'r-', linewidth=2, label='HW slack')
    ax2.plot(series_X, lstm_slack, 'b--', linewidth=2, label='LSTM slack')
    
    

    t2 = ("Container " + contname)
    fig2.suptitle(t2, fontsize=23)
    ax2.tick_params(axis="x", labelsize=20) 
    ax2.tick_params(axis="y", labelsize=20) 
    fig2.subplots_adjust(left=0.1, bottom=0.2, right=None, top=None, wspace=None, hspace=None)

    leg2 = ax2.legend(loc='lower center', bbox_to_anchor=(0.5, -0.30), fancybox=True, shadow=True, ncol=6, fontsize=15)
    leg2_lines = leg2.get_lines()
    plt.setp(leg2_lines, linewidth=5)

    ax2.set_xlabel('Observations', fontsize=20)
    ax2.set_ylabel('CPU (millicores)', fontsize=20)
    # ax2.set_ylim(bottom=-100)
    # ax2.set_ylim(top=505)

    # PLOT 3 Requests

    ax3.plot(series_X, series, 'y-', linewidth=1,label='CPU usage')
    ax3.plot(series_X, vpa_target, 'g--', linewidth=1,label='VPA requested')
    ax3.plot(X_requests,hw_requests, 'r-', linewidth=2,label='HW requested')
    ax3.plot(X_requests,lstm_requests, 'b--', linewidth=2,label='LSTM requested')
    
    

    t3 = ("Container " + contname)
    fig3.suptitle(t3, fontsize=23)
    ax3.tick_params(axis="x", labelsize=20) 
    ax3.tick_params(axis="y", labelsize=20) 
    fig3.subplots_adjust(left=0.1, bottom=0.2, right=None, top=None, wspace=None, hspace=None)
    leg3 = ax3.legend(loc='lower center', bbox_to_anchor=(0.5, -0.30), fancybox=True, shadow=True, ncol=6, fontsize=15)
    leg3_lines = leg3.get_lines()
    plt.setp(leg3_lines, linewidth=5)

    ax3.set_xlabel('Observations', fontsize=20)
    ax3.set_ylabel('CPU (millicores)', fontsize=20)


    # Print out 

    # print("---AVERAGE SLACK---")
    skip = params["season_len"]
    lstm_requests.pop()
    hw_requests.pop()

    lstm_reqs = lstm_requests[skip*2:]
    hw_reqs = hw_requests[skip*2:]
    cpu_usages = series[skip*2:]
    vpa_reqs = vpa_target[skip*2:]
    

    
    lstm_slack = np.subtract(lstm_reqs,cpu_usages)
    hw_slack = np.subtract(hw_reqs,cpu_usages)
    vpa_slack = np.subtract(vpa_reqs,cpu_usages)

    slack_list_vpa.append(vpa_slack)
    slack_list_hw.append(hw_slack)
    slack_list_lstm.append(lstm_slack)

    # Average slack print out

    print("avg HW:", np.mean(slack_list_hw))
    print("avg LSTM:", np.mean(slack_list_lstm))
    print("avg VPA:", np.mean(slack_list_vpa))

    print("---% TIME ABOVE REQUESTED---")
    lstm_count = 0
    lstm_total = 0
    hw_count = 0
    hw_total = 0
    vpa_count = 0
    vpa_total = 0

    cpu_above_lstm = []
    cpu_above_hw = []
    cpu_above_vpa = []

    for i in range(len(cpu_usages)):
        if cpu_usages[i] > lstm_reqs[i]:
            lstm_count += 1
            amount = cpu_usages[i] - lstm_reqs[i]
            lstm_total += cpu_usages[i] - lstm_reqs[i]
            cpu_above_lstm.append(amount)
        if cpu_usages[i] > hw_reqs[i]:
            hw_count += 1
            amount = cpu_usages[i] - hw_reqs[i]
            hw_total += cpu_usages[i] - hw_reqs[i]
            cpu_above_hw.append(amount)
        if cpu_usages[i] > vpa_reqs[i]:
            vpa_count += 1
            amount = cpu_usages[i] - vpa_reqs[i]
            vpa_total += cpu_usages[i] - vpa_reqs[i] 
            cpu_above_vpa.append(amount)

    above_list_hw.append(cpu_above_hw)
    above_list_lstm.append(cpu_above_lstm)
    above_list_vpa.append(cpu_above_vpa)
    
    perc_above_lstm = round(lstm_count/len(cpu_usages), 4)*100
    perc_above_hw = round(hw_count/len(cpu_usages), 4)*100
    perc_above_vpa = round(vpa_count/len(cpu_usages), 4)*100

    
    perc_list_vpa.append(perc_above_vpa)
    perc_list_lstm.append(perc_above_lstm)
    perc_list_hw.append(perc_above_hw)
    
    # print("Usage above requested LSTM: " + str(perc_above_lstm) + "%")
    # print("Usage above requested HW: " + str(perc_above_hw) + "%")
    # print("Usage above vpa_target: " + str(perc_above_vpa) + "%")

    # BOX 1: Slacks
    # print("Slack list vpa: ", slack_list_vpa)
    # print("Slack list hw: ", slack_list_hw)
    # print("Slack list lstm: ", slack_list_lstm)

    # # BOX 2: CPU above values 
    # print("CPU above lstm: ", above_list_lstm)
    # print("CPU above hw: ", above_list_hw)
    # print("CPU above vpa: ", above_list_vpa)

    # HELP plot 1: %observations above
    print("perc above vpa_target: ", perc_list_vpa)
    print("perc above hw: ", perc_list_hw)
    print("perc above lstm: ", perc_list_lstm)

    # ax1.set_xlim(left=params["season_len"]*2)
    # ax2.set_xlim(left=params["season_len"]*2)

    fig1.set_size_inches(15,8)
    fig2.set_size_inches(15,8)
    fig3.set_size_inches(15,8)
    # plt.show()

    fig1.savefig("./pred"+contname+".png",bbox_inches='tight')
    fig2.savefig("./slack"+contname+".png", bbox_inches="tight")  
    fig3.savefig("./scale"+contname+".png", bbox_inches="tight")  

    ax1.clear() 
    ax2.clear()
    ax3.clear()
    

    # if vpa_count == 0:
    #     above_list_vpa.append(0)
    # else:
    #     above_list_vpa.append(vpa_total/vpa_count)
    # if lstm_count == 0:
    #     above_list_hw.append(0)
    # else:
    #     above_list_hw.append(lstm_total/lstm_count)

    # print("avg above: ", above_list_vpa)
    # print("avg above hw: ", above_list_hw)

def set_box_color(bp, color):
    plt.setp(bp['boxes'], color=color, linewidth=2)
    plt.setp(bp['whiskers'], color=color, linewidth=2)
    plt.setp(bp['caps'], color=color, linewidth=2)
    plt.setp(bp['medians'], color=color, linewidth=2)

if __name__ == '__main__':
    
    std = 300
    alpha = 0.7
    main()
   
   

