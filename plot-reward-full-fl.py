import pickle
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
plt.rcParams.update({'font.size': 18})

# out_name = "out_frozen_lake3.csv"  # Default: out.csv
hist_name = 'reward-pg-raw-fl.pkl'

# DF = pd.read_csv(out_name)
# DF.rename(columns={'Errors': 'Asymptotic Convergence Error'}, inplace=True)

with open(hist_name, "rb") as f:  # Python 3: open(..., 'rb')
    r1, r2, r3 = pickle.load(f)
f.close()


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


easy_plot(r1, "g", "Policy Gradient", cut_off=300)
easy_plot(r2, "orange", "Greedy-GQ", cut_off=300)
easy_plot(r3, "b", "VR-Greedy-GQ: M=3000" , cut_off=300)

plt.setp(ax1, xticks=[0,   100, 200, 300], xticklabels=['0', '100k', '200k', '300k' ] )
#ax1.set_ylim(-1, 75)

#ax1.legend(loc="lower right")
ax1.legend(loc='upper center', bbox_to_anchor=(0.5, 1.3),
          ncol=1, fancybox=True, shadow=True)
ax1.set_ylabel(r"Maximum Reward")
ax1.set_xlabel("# of iterations")
fig.tight_layout()
# fig.set_size_inches(10, 10, forward=True)
#fig.savefig("fig-fl.png", dpi=300)
plt.show()
