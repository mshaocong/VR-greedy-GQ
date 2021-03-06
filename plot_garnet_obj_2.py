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
with open('reward-gq-alpha002-beta001-bs3000-seed123-max.pkl', 'rb') as f:  # Python 3: open(..., 'wb')
    r1 = np.array(pickle.load(f))
f.close()
with open('reward-vrgq-alpha002-beta001-bs3000-seed123-max.pkl', 'rb') as f:  # Python 3: open(..., 'wb')
    r2 = np.array(pickle.load(f))
f.close()
with open('obj-gq-alpha002-beta001-bs3000-seed123-min.pkl', 'rb') as f:  # Python 3: open(..., 'wb')
    gq_obj = np.array(pickle.load(f))
f.close()
with open('obj-vrgq-alpha002-beta001-bs3000-seed123-min.pkl', 'rb') as f:  # Python 3: open(..., 'wb')
    vrgq_obj = np.array(pickle.load(f))
f.close()


# sns.set(style="ticks", palette="pastel")
fig, ax1 = plt.subplots(figsize=(7, 6))


def easy_plot(hist, color, label, cut_off=None, percentile=90, fill=True):
    upper_loss = np.percentile(hist, percentile, axis=0)
    lower_loss = np.percentile(hist, 100 - percentile, axis=0)
    avg_loss = np.mean(hist, axis=0)
    x = np.arange(avg_loss.shape[0])

    if cut_off is None:
        ax1.plot(x[:cut_off], avg_loss, c=color, label=label)
    else:
        ax1.plot(x[:cut_off], list(avg_loss[:cut_off]), c=color, label=label)

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



easy_plot(gq_obj, "orange", "Greedy-GQ", cut_off=100)
easy_plot2(vrgq_obj, "b", "VR-Greedy-GQ: M=3000" , cut_off=100)
plt.setp(ax1, xticks=[0, 20, 40, 60, 80, 100], xticklabels=['0', '2k', '4k', '6k', '8k', '10k'] )
# ax1.set_ylim(-0.5, 10.0)

ax1.legend(loc='upper right')
ax1.set_ylabel(r"$\min\ J(\theta)$")
ax1.set_xlabel("# of gradient computations")
fig.tight_layout()
# fig.set_size_inches(10, 10, forward=True)
# fig.savefig("fig-fl.png", dpi=300)
plt.show()
