import pickle
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
plt.rcParams.update({'font.size': 18})

# out_name = "out_frozen_lake3.csv"  # Default: out.csv
hist_name = 'reward-pg-raw.pkl'

# DF = pd.read_csv(out_name)
# DF.rename(columns={'Errors': 'Asymptotic Convergence Error'}, inplace=True)

with open(hist_name, "rb") as f:  # Python 3: open(..., 'rb')
    r1, r2, r3 = pickle.load(f)
f.close()

#with open(hist_name, "rb") as f:  # Python 3: open(..., 'rb')
#    hist_vrgq = pickle.load(f)
#f.close()
# Covert number of iterations to Number of gradients
hist_main = []
for hist_ in r3:
    hist_tmp = []
    for i in range(len(hist_)):
        if i % 300 == 0:
            hist_tmp = hist_tmp + [hist_[i]] * 300
        else:
            hist_tmp.append(hist_[i])
            hist_tmp.append(hist_[i])
    hist_main.append(hist_tmp[:len(r2[0]) ])
r3 = hist_main

# sns.set(style="ticks", palette="pastel")
fig, ax1 = plt.subplots(figsize=(7, 6))

def easy_plot(hist, color, label, cut_off=None, percentile=90, fill=True):
    upper_loss = np.percentile(hist, percentile, axis=0)
    lower_loss = np.percentile(hist, 100 - percentile, axis=0)
    avg_loss = np.mean(hist, axis=0)
    x = np.arange(avg_loss.shape[0])

    if cut_off is None:
        ax1.plot(avg_loss, c=color, label=label)
    else:
        ax1.plot(list(avg_loss[:cut_off]), c=color, label=label)

    if fill:
        if cut_off is None:
            ax1.fill_between(x[:cut_off], lower_loss[:cut_off], upper_loss[:cut_off], color=color, alpha=0.3)
        else:
            ax1.fill_between(x[:cut_off], lower_loss[:cut_off], upper_loss[:cut_off], color=color, alpha=0.3)


easy_plot(r2, "orange", "Greedy-GQ", cut_off=1500)
easy_plot(r3, "b", "VR-Greedy-GQ: M=3000" , cut_off=1500)

upper_r = np.percentile(r1, 90, axis=0)
lower_r = np.percentile(r1, 100 - 90, axis=0)
avg_r = np.mean(r1, axis=0)
x = np.arange(0, avg_r.shape[0]*180, 180)
ax1.plot(x[:9], avg_r[:9] ,c="g", label="Policy Gradient")
ax1.fill_between(x[:9], lower_r[:9], upper_r[:9], color="g",alpha=0.3)

plt.setp(ax1, xticks=[0, 300, 600, 900, 1200, 1500], xticklabels=['0', '3k', '6k', '9k', '12k', '15k'] )
#ax1.set_ylim(-1, 75)

ax1.legend(loc="lower right")
ax1.set_ylabel(r"Maximum Reward")
ax1.set_xlabel("# of samples")
fig.tight_layout()
# fig.set_size_inches(10, 10, forward=True)
#fig.savefig("fig-fl.png", dpi=300)
plt.show()
