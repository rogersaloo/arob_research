import config
lr = config.LEARNING_RATE
batch = config.BATCH_SIZE
epoch = config.NUM_EPOCHS
 #Batch 1-ep_100 lr1e-2
# NORMAL_PATH = "fid/batch_4-epoch_100-lr_0.001_0/real"
# SYNTHESIS_PATH = "fid/batch_4-epoch_100-lr_0.001_0/synthetic/"
# NORMAL_IMAGE_LABEL="norm_lr1e-2_ep100_b1"
# PNEUMONIA_IMAGE_LABEL="pneu_lr1e-2_ep100_b1"

# new_path = os.path.join(path,fid_config.NORMAL_PATH)
# print(new_path)
# sys.exit()
import config


#Batch 1-ep_100 lr1e-2
NORMAL_PATH = f"fid/batch_{batch}-epoch_{epoch}-lr_{lr}_0/real"
SYNTHESIS_PATH = f"fid/batch_{batch}-epoch_{epoch}-lr_{lr}_0/synthetic/"
NORMAL_IMAGE_LABEL=f"norm-batch_{batch}-epoch_{epoch}-lr_{lr}"
PNEUMONIA_IMAGE_LABEL=f"pneu-batch_{batch}-epoch_{epoch}-lr_{lr}"

#  #Batch 1-ep_100 lr1e-5
# NORMAL_PATH = "fid/bath_1-epoch100_size-lr3e-3/real"
# SYNTHESIS_PATH = "fid/bath_1-epoch100_size-lr3e-3/synthetic/"
# NORMAL_IMAGE_LABEL="norm_lr1e-3_ep100_b1"
# PNEUMONIA_IMAGE_LABEL="pneu_lr1e-3_ep100_b1"

# #Batch 1-ep_100 lr1e-5
# NORMAL_PATH = "fid/batch_1-l1e-5_ep100_b1/real"
# SYNTHESIS_PATH = "fid/batch_1-l1e-5_ep100_b1/synthetic/"
# NORMAL_IMAGE_LABEL="norm_lr1e-3_ep100_b1"
# PNEUMONIA_IMAGE_LABEL="pneu_lr1e-3_ep100_b1"

#Batch 100 lr1e-4 real
# NORMAL_PATH = "fid/bath_1-epoch100_size-lr1e-4/real"
# NORMAL_PATH = "fid/bath_1-epoch100_size-lr1e-4/real"

#Batch 100 lr3e-4 real
# NORMAL_PATH = "fid/bath_1-epoch100_size-lr3e-4/real"
# SYNTHESIS_PATH = "fid/bath_1-epoch100_size-lr3e-4/synthetic/"

#Batch 100 lr3e-6 real
# NORMAL_PATH = "fid/bath_1-epoch100_size-lr3e-6/real"
# SYNTHESIS_PATH = "fid/bath_1-epoch100_size-lr3e-6/synthetic/"

#Batch 100 lr3e-6 real
# NORMAL_PATH = "fid/bath_1-epoch100_size-lr3e-6/real"
# SYNTHESIS_PATH = "fid/bath_1-epoch100_size-lr3e-6/synthetic/"
# NORMAL_IMAGE_LABEL="norm_l1e-6_ep100_b8"
# PNEUMONIA_IMAGE_LABEL="pneu_l1e-6_ep100_b8"

# NORMAL_PATH = "fid/bath_1-epoch100_size-lr1e-5/real"
# SYNTHESIS_PATH = "fid/bath_1-epoch100_size-lr1e-5/synthetic/"
# NORMAL_IMAGE_LABEL="norm_l1e-5_ep100_b1"
# PNEUMONIA_IMAGE_LABEL="pneu_l1e-5_ep100_b1