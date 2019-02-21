import numpy as np
import data_load

##### Data Loader
DataLoader=data_load.DataLoader()

##### training set
# DataLoader.set_data_path(is_train=True)
# DataLoader.save_total_motion(18, 24, 8)
# input_batch, label_batch = DataLoader.get_total_motion(18, 24, 8)
# np.save('input_batch2.npy', input_batch)
# np.save('label_batch2.npy', label_batch)

#### test set
# DataLoader.set_data_path(is_train=False)
# DataLoader.save_total_motion(18, 24)
# test_input_batch, test_label_batch = DataLoader.get_total_motion(18, 24)
# np.save('test_input_batch2.npy', test_input_batch)
# np.save('test_label_batch2.npy', test_label_batch)

total_input_batch, total_label_batch = DataLoader.merge_velocity_data()
data_len= len(total_input_batch)
train_len = int(data_len * 0.9)
np.save('input_batch3.npy', total_input_batch[:train_len])
np.save('label_batch3.npy', total_label_batch[:train_len])
np.save('test_input_batch3.npy', total_input_batch[train_len:])
np.save('test_label_batch3.npy', total_label_batch[train_len:])