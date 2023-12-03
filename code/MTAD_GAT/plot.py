from matplotlib import pyplot as plt
from data_config import *
import pickle
save_dir = exp_dir / 'plot'
save_dir.mkdir(parents=True, exist_ok=True)
data_index = 28
train_data_item, test_data_item = get_data_by_index(dataset_type, data_index, training_period)
x_train, x_test = preprocess(train_data_item.T, test_data_item.T)


test_output = pickle.load(open(exp_dir / 'result/{}/test_output.pkl'.format(data_index),'br'))
y_test = label[data_index]
print(sum(y_test))
print(x_test.shape)
print(test_output['Recon_0'].shape)
fig = plt.figure(figsize=(30,60))
_len = x_test.shape[1]
anomaly_score=test_output['A_Score_Global']
print(anomaly_score.shape)
for i in range(_len):
    ax = fig.add_subplot(_len+1,1,i+2)
    recon = test_output[f'Recon_{i}']
    plot_x = len(recon)
    data = x_test[-plot_x:,i]
    ax.plot(range(plot_x),data,color='blue')
    ax.plot(range(plot_x),recon,color='green')
ax = fig.add_subplot(_len+1,1,1)
ax.plot(range(plot_x),anomaly_score,color='red')
plt.savefig(save_dir/f"{data_index}.png")
