import pickle
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
sns.set(rc={'figure.figsize':(6,8)})

with open( 'hist-error--2.pkl', "rb") as f:  # Python 3: open(..., 'rb')
    hist_1000= pickle.load(f)
f.close()
with open( 'hist-error-1000-2.pkl', "rb") as f:  # Python 3: open(..., 'rb')
    hist_2000= pickle.load(f)
f.close()
with open( 'hist-error-2000-2.pkl', "rb") as f:  # Python 3: open(..., 'rb')
    hist_3000= pickle.load(f)
f.close()
with open( 'hist-error-3000-2.pkl', "rb") as f:  # Python 3: open(..., 'rb')
    hist_4000= pickle.load(f)
f.close()


var_h = [hist_1000 , hist_2000, hist_3000, hist_4000]
num_obs = 20000

errors_vrgq = [np.mean(np.array(h)[:, -num_obs:], axis=1) for h in var_h]

batch_size_list = [" 1", "1000", "2000", "3000"]
DF = pd.DataFrame(columns=["Errors", "Batch Size"])
count = 0
for i in range(len(batch_size_list)):
    num_simulation = 60
    for j in range(num_simulation):
        DF.loc[count] = [errors_vrgq[i][j], batch_size_list[i]]
        count += 1

sns.set(style="ticks", palette="pastel")
bp = sns.boxplot(x="Batch Size", y="Errors", data=DF)
fig = bp.get_figure()
bp.set_xlabel("Batch Size",fontsize=18)
bp.set_ylabel("Errors",fontsize=18)
bp.tick_params(labelsize=18)

fig.savefig("box.png", dpi=300)
