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
with open('reward-gq-alpha02-beta01-bs3000-seed114514-max-all.pkl', 'rb') as f:  # Python 3: open(..., 'wb')
    hist = np.array(pickle.load(f))
f.close()

gq_obj = [h for h in hist[0]]
vrgq_obj = [h for h in hist[1]]

gq_obj = np.array(gq_obj)
vrgq_obj = np.array(vrgq_obj)

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


easy_plot(gq_obj, "orange", "Greedy-GQ", cut_off=500)
easy_plot(vrgq_obj, "b", "VR-Greedy-GQ: M=3000" , cut_off=500)
plt.setp(ax1, xticks=[0, 100, 200, 300, 400, 500], xticklabels=['0', '100k', '200k', '300k', '400k', '500k'] )
ax1.set_ylim(-0.0005, 0.01)

ax1.legend(loc='lower right')
ax1.set_ylabel(r"Maximum Reward")
ax1.set_xlabel("# of iterations")
fig.tight_layout()
# fig.set_size_inches(10, 10, forward=True)
# fig.savefig("fig-fl.png", dpi=300)
plt.show()
