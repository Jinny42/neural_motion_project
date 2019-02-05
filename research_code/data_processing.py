import numpy as np
import data_load

##### Data Loader
DataLoader=data_load.DataLoader()

##### training set
DataLoader.set_data_path(is_train=True)
input_batch, label_batch = DataLoader.get_total_motion(0, 3, 8)

#### test set
DataLoader.set_data_path(is_train=False)
test_input_batch, test_label_batch = DataLoader.get_total_motion(0, 3)

np.save('input_batch.npy', input_batch)
np.save('label_batch.npy', label_batch)
np.save('test_input_batch.npy', test_input_batch)
np.save('test_label_batch.npy', test_label_batch)