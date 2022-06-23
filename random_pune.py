import random
import numpy as np

remain_num = 35000
indice = random.sample([i for i in range(50000)], remain_num)

savedir = 'random_pune.txt'
np.savetxt(savedir ,np.array(indice))