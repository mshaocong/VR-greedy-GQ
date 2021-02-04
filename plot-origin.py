import pickle
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
plt.rcParams.update({'font.size': 18})

# out_name = "out_frozen_lake3.csv"  # Default: out.csv
hist_name = 'hist-fl-44.pkl'

# DF = pd.read_csv(out_name)
# DF.rename(columns={'Errors': 'Asymptotic Convergence Error'}, inplace=True)

with open(hist_name, "rb") as f:  # Python 3: open(..., 'rb')
    hist_gq, hist_vrgq = pickle.load(f)
f.close()
hist_name = 'hist-error-min-3000.pkl'

#with open(hist_name, "rb") as f:  # Python 3: open(..., 'rb')
#    hist_vrgq = pickle.load(f)
#f.close()
# Covert number of iterations to Number of gradients
hist_main = []
for hist_ in hist_vrgq:
    hist_tmp = []
    for i in range(len(hist_)):
        if i % 3 == 0:
            hist_tmp = hist_tmp + [hist_[i]] * 3
        else:
            hist_tmp.append(hist_[i])
            hist_tmp.append(hist_[i])
    hist_main.append(hist_tmp[:len(hist_gq[0]) ])
hist_vrgq = hist_main

# sns.set(style="ticks", palette="pastel")
fig, ax1 = plt.subplots(figsize=(6, 6))

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


easy_plot(hist_gq, "orange", "Greedy-GQ", cut_off=100)
easy_plot(np.array(hist_vrgq), "b", "VR-Greedy-GQ: M=3000" , cut_off=100)
plt.setp(ax1, xticks=[0, 20, 40, 60, 80, 100], xticklabels=['0', '20k', '40k', '60k', '80k', '100k'] )
ax1.set_ylim(-1, 75)

ax1.legend(loc=1)
ax1.set_ylabel(r"$\min ||\nabla J(\theta)||^2$")
ax1.set_xlabel("# of gradient computations")
fig.tight_layout()
# fig.set_size_inches(10, 10, forward=True)
fig.savefig("fig-fl.png", dpi=300)
plt.show()
