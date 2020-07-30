
# l = [1, 'a']
#
# for i in l:
#     if i + 1 >= 2:
#         print('mayor que 2')
#     else:
#         print('menor que 2')
import statistics
import numpy as np

l1 = [1,2,3,5,3,None,None,4,6,3.4]
l2 = np.array(l1, dtype=np.float64)
print(l2)
print(np.nanmax(l2))
import matplotlib.pyplot as plt

plt.plot(range(len(l1)), l1)
plt.show()