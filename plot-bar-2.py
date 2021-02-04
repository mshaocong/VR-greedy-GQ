import pickle
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
plt.rcParams.update({'font.size': 18})
with open( 'hist-fl-44.pkl', "rb") as f:  # Python 3: open(..., 'rb')
    hist_1000, _ = pickle.load(f)
f.close()
with open( 'hist-error-200-fl.pkl', "rb") as f:  # Python 3: open(..., 'rb')
    hist_2000= pickle.load(f)
f.close()
with open( 'hist-error-500-fl.pkl', "rb") as f:  # Python 3: open(..., 'rb')
    hist_3000= pickle.load(f)
f.close()
with open( 'hist-error-1000-fl.pkl', "rb") as f:  # Python 3: open(..., 'rb')
    hist_4000= pickle.load(f)
f.close()

var_h = [hist_1000 , hist_2000, hist_3000, hist_4000]
num_obs = 20

errors_vrgq = [np.mean(np.array(h)[:, -num_obs:], axis=1) for h in var_h]
errors_vrgq = [np.mean(hist) for hist in errors_vrgq]

batch_size_list = [" 1", "200", "400", "800"]

# import matplotlib.pyplot as plt

plt.bar([0,1,2,3], errors_vrgq, tick_label = batch_size_list)
plt.xlabel("Batch Size")
plt.ylabel("Convergence Error")
plt.title("Convergence Error of VR-Greedy-GQ \n under Different Batch Sizes")
plt.ylim(0, 1.0)
plt.savefig("bar.png", dpi=300)
plt.show()