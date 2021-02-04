import pickle
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
plt.rcParams.update({'font.size': 18})

# out_name = "out_frozen_lake3.csv"  # Default: out.csv
hist_name = 'garnet-var-60.pkl'

# DF = pd.read_csv(out_name)
# DF.rename(columns={'Errors': 'Asymptotic Convergence Error'}, inplace=True)

with open(hist_name, "rb") as f:  # Python 3: open(..., 'rb')
    hist_gq, hist_vrgq = pickle.load(f)
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


# easy_plot(hist_gq, "orange", "Greedy-GQ", cut_off=len(hist_vrgq[0]))
# easy_plot(np.array(hist_vrgq), "b", "VR-Greedy-GQ: M=3000" )
easy_plot(hist_gq, "orange", "Greedy-GQ", cut_off=400)
easy_plot(np.array(hist_vrgq), "b", "VR-Greedy-GQ: M=3000", cut_off=400)

#ax1.set_ylim(-0.05, 2.0)
plt.setp(ax1, xticks=[0, 50, 100, 150, 200], xticklabels=['0', '50k', '100k', "150k", "200k"] )
ax1.legend(loc=1)
ax1.set_ylabel(r"Estimated Gradient Variance")
ax1.set_xlabel("# of iterations")
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
fig.tight_layout()
# fig.set_size_inches(10, 8, forward=True)
fig.savefig("fig-var-fl.png", dpi=300)
plt.show()
