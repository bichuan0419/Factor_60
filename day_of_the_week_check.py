import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# load stuff
currFolder = os.getcwd()
file_path = os.path.join(currFolder, 'log_data/log_file.csv')
col_names = ['trddt','market_cap']
log_file = pd.read_csv(file_path,index_col=0,names=col_names)
log_file.index = pd.to_datetime(log_file.index)
vec_weekday = []


# transformations
for i in range(len(log_file)):
    vec_weekday.append(log_file.index[i].weekday())
log_file['Weekday'] = vec_weekday
return_hist = log_file['market_cap']
day_diff = np.zeros(len(return_hist))
for i in range(len(return_hist)-1):
    day_diff[i+1] = (return_hist[i + 1] - return_hist[i])/return_hist[i]
log_file['day_diff'] = day_diff


# plot stuff
fig, axes = plt.subplots(5,1, figsize = (5,11), sharex=True)

for i in range(5):
    log_file.loc[log_file['Weekday'] == i]['day_diff'].plot.hist(bins = 20,ax=axes[i])
    print("mean value:", np.mean(log_file.loc[log_file['Weekday'] == i]['day_diff']))
    print("variance:", np.var(log_file.loc[log_file['Weekday'] == i]['day_diff']))
    # fig = plt.gcf()
    #     fig.set_size_inches(18,10)
    axes[i].set_title('Tot_rank29, Weekday: %i' %i)
    # plt.xlim(-0.1,0.1)
# plt.show()

# # save figure
file_path = os.path.join(currFolder, 'figures/log_file_DOW.png')
fig.savefig(file_path)