import pickle
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({'font.size': 18})

# out_name = "out_frozen_lake3.csv"  # Default: out.csv
hist_name = 'reward-ac-raw.pkl'

# DF = pd.read_csv(out_name)
# DF.rename(columns={'Errors': 'Asymptotic Convergence Error'}, inplace=True)

with open(hist_name, "rb") as f:  # Python 3: open(..., 'rb')
    r1, r2, r3 = pickle.load(f)
f.close()

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


def easy_plot2(hist, color, label, cut_off=None, percentile=90, fill=True):
    upper_loss = np.percentile(hist, percentile, axis=0)
    lower_loss = np.percentile(hist, 100 - percentile, axis=0)
    avg_loss = np.mean(hist, axis=0)
    x = np.arange(avg_loss.shape[0])
    new_x = x * 3
    if cut_off is None:
        raise NotImplementedError
    else:
        ax1.plot(new_x[:1+int(np.floor(cut_off/3))], list(avg_loss[:1+int(np.floor(cut_off/3))]), c=color, label=label)

    if fill:
        if cut_off is None:
            raise NotImplementedError
        else:
            ax1.fill_between(new_x[:1+int(np.floor(cut_off/3))], lower_loss[:1+int(np.floor(cut_off/3))], upper_loss[:1+int(np.floor(cut_off/3))], color=color, alpha=0.3)


def easy_plot3(hist, color, label, cut_off=None, percentile=90, fill=True):
    upper_loss = np.percentile(hist, percentile, axis=0)
    lower_loss = np.percentile(hist, 100 - percentile, axis=0)
    avg_loss = np.mean(hist, axis=0)
    x = np.arange(  avg_loss.shape[0]  )
    new_x = np.arange(0, avg_loss.shape[0], 1000)
    if cut_off is None:
        raise NotImplementedError
    else:
        ax1.plot(x[:cut_off+1], np.take(avg_loss, new_x), c=color, label=label)

    if fill:
        if cut_off is None:
            raise NotImplementedError
        else:
            ax1.fill_between(x[:cut_off+1], np.take(lower_loss, new_x), np.take(upper_loss, new_x), color=color, alpha=0.3)




easy_plot(r1, "g", "Actor-Critic", cut_off=1500)
easy_plot(r2, "orange", "Greedy-GQ", cut_off=1500)
easy_plot2(r3, "b", "VR-Greedy-GQ: M=3000" , cut_off=1500)

plt.setp(ax1, xticks=[0, 300, 600, 900, 1200, 1500], xticklabels=['0', '3k', '6k', '9k', '12k', '15k'] )
#ax1.set_ylim(-1, 75)

# ax1.legend(loc="lower right")

ax1.legend(loc='upper center', bbox_to_anchor=(0.5, 1.4),
          ncol=1, fancybox=True, shadow=True)
ax1.set_ylabel(r"Maximum Reward")
ax1.set_xlabel("# of gradient computations")
fig.tight_layout()
#plt.yscale("log")
# fig.set_size_inches(10, 10, forward=True)
#fig.savefig("fig-fl.png", dpi=300)
plt.show()
