import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()

n = range(1, 1001)
beta_0 = 10 
beta_1 = 5
beta_list = []
beta_int = []

for i in range(1, 1001):
    x_i = np.random.normal(0, 5, 1000)
    e_i = np.random.normal(0, 5, 1000)

    y_i = beta_0 + beta_1*x_i + e_i

    x_array = np.array([x_i]).T
    x_array_T = np.array([x_i])
    y_array = np.array([y_i]).T

    inverse = np.linalg.inv(np.dot(x_array_T, x_array))
    beta = np.linalg.multi_dot([inverse, x_array_T, y_array])
    beta_list.append(beta)

for i in range(0, 1000):
    beta_int.append(beta_list[i][0][0])

beta_df = pd.DataFrame(data = beta_int, index = range(1, 1001), columns = ['beta'])
mean_beta = np.mean(beta_df['beta'])

sns.histplot(beta_df, x = 'beta')
plt.axvline(x = mean_beta, color = 'firebrick', linestyle = '--')

plt.title('Beta estimation histogram 1000 observations')
plt.xlabel('Beta estimator')
plt.ylabel('Count')

plt.show()
