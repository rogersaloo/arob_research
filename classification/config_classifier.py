
#Variables
# csv_file = "train_metadata.csv",
csv_file = '../sample_data/sample_data_combine.csv'
# root_dir = "alldata/train/real_train_images",
root_dir = '../sample_data/sample_images_combined'
loss_tensorboard = 'runsclassification/loss'
accuracy_tensorboard = 'runsclassification/accuracy'

# Hyperparameters
in_channel = 1
num_classes = 2
learning_rate = 1e-3
batch_size = 16
num_epochs = 4