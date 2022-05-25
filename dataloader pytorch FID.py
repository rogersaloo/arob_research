from dataset import NormalPneumonia
import numpy as np


from torchvision import datasets, transforms
unnormalize = lambda x: x / 2.0 + 0.5
to_nhwc = lambda x: np.transpose(x, (0, 2, 3, 1))
dataset = datasets.CIFAR10(root='./training_images', download=True,
                           transform=transforms.Compose([
                               transforms.Resize(32),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))
dataset_full = np.array([x[0].cpu().numpy() for x in iter(dataset)])
dataset_nhwc = np.clip(255 * to_nhwc(unnormalize(dataset_full)), 0.0, 255)

print('fid with pytorch dataset')
fid2 = fid_score(create_session, dataset_nhwc, gen_samples_list)
print(fid2)
# 29.85279935346972
assert fid2 < 32