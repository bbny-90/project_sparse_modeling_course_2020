import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize
import cvxpy as cvx
in_folder = "2d_array_images/"
num_total_data = 300
lamda = .01
np.random.seed(2)

if 0:
    all_data_downsampled = []
    for i in range(num_total_data):
        img = np.load(in_folder + "{}.npy".format(i))
        assert np.allclose(img.shape, np.array([265, 265]))
        # plt.matshow(img)
        image_resized = resize(img, (img.shape[0] // 5, img.shape[1] // 5),
                       anti_aliasing=True)
        all_data_downsampled.append(image_resized.flatten())
    all_data_downsampled = np.stack(all_data_downsampled)
    with open(in_folder + 'all_downsampled.npy', 'wb') as f:
        np.save(f, all_data_downsampled)
else:
    all_data_downsampled = np.load(in_folder + "all_downsampled.npy")

_, num_pix = all_data_downsampled.shape
meas_indx = np.random.choice(num_pix, int(num_pix*0.05), replace=False)
if 0:
    for i in range(1, 300):
        plt.matshow(all_data_downsampled[i, :].reshape(53, 53))
        # meas_x, meas_y = np.unravel_index(meas_indx, (53,53))
        # plt.plot(meas_x, meas_y, 'ro')
        plt.show()
        plt.close()
    exit()


rand_indx = np.arange(num_total_data)
np.random.shuffle(rand_indx)
train_indx = int(0.8*num_total_data)

# mean = np.mean(all_data_downsampled, axis=0)
# std = np.std(all_data_downsampled, axis=0)
# all_data_downsampled -= mean
# all_data_downsampled = all_data_downsampled / std
all_data_downsampled /= np.max(all_data_downsampled, axis=0)

train_data = all_data_downsampled[0:train_indx, meas_indx]
target_field_id = train_indx+ 10
y = all_data_downsampled[target_field_id, meas_indx]


w = cvx.Variable(train_data.shape[0])
loss = cvx.sum_squares(train_data.T @ w - y )/2 + lamda * cvx.norm(w,1)

problem = cvx.Problem(cvx.Minimize(loss))
problem.solve(verbose=True) 
opt = problem.value
print('Optimal Objective function value is: {}'.format(opt))
print(w.value)