import pickle
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
plt.rcParams.update({'font.size': 18})
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
errors_vrgq = [np.mean(hist) for hist in errors_vrgq]

batch_size_list = [" 1", "1000", "2000", "3000"]

# import matplotlib.pyplot as plt

plt.bar([0,1,2,3], errors_vrgq, tick_label = batch_size_list)
plt.savefig("bar.png", dpi=300)
plt.xlabel("Batch Size")
plt.ylabel("Convergence Error")
#plt.ylim(0,0.02)
plt.title("Convergence Error of VR-Greedy-GQ \n under Different Batch Sizes")
# plt.figure(figsize=(10,20))
plt.show()