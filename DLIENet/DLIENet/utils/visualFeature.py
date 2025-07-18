import os
import argparse
import random

import torch
import torchvision
from matplotlib import pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader
from collections import OrderedDict
from utils import write_img, chw_to_hwc
from datasets.loader import SingleLoader
from models import *
from torchvision.io.image import read_image
from torchvision.transforms.functional import normalize, resize, to_pil_image
from torchvision.models import resnet152
from torchcam.methods import SSCAM
import matplotlib.pyplot as plt
from torchcam.utils import overlay_mask

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='MixDehazeNet-b', type=str, help='model name')
parser.add_argument('--num_workers', default=16, type=int, help='number of workers')
parser.add_argument('--data_dir', default='./data/', type=str, help='path to dataset')
parser.add_argument('--save_dir', default='./saved_models/indoor/', type=str, help='path to models saving')
parser.add_argument('--result_dir', default='./results/', type=str, help='path to results saving')
parser.add_argument('--folder', default='RESIDE-IN/test', type=str, help='folder name')
parser.add_argument('--exp', default='indoor', type=str, help='experiment setting')
args = parser.parse_args()


def single(save_dir):
	state_dict = torch.load(save_dir)['state_dict']
	new_state_dict = OrderedDict()

	for k, v in state_dict.items():
		name = k[7:]
		new_state_dict[name] = v

	return new_state_dict

def test(test_loader, network, result_dir):
    torch.cuda.empty_cache()

    network.eval()

    # network = torchvision.models._utils.IntermediateLayerGetter(network, {'patch_embed': '1', 'layer1': '2'})

    os.makedirs(result_dir, exist_ok=True)

    for batch in tqdm(test_loader):
        input = batch['img'].cuda()
        print(input.size())
        filename = batch['filename'][0]

        with torch.no_grad():
            img = read_image("/home/xq/Project/DehazeFormer-main/test/model/1.jpg")
            # Preprocess it for your chosen model
            # input_tensor = normalize(resize(img, (224, 224)) / 255., [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]).cuda()
            # print(input_tensor.size())
            # with SmoothGradCAMpp(model) as cam_extractor:
            cam_extractor = SSCAM(network)
            # Preprocess your data and feed it to the model
            out = network(input)
            # Retrieve the CAM by passing the class index and the model output
            activation_map = cam_extractor(out.squeeze(0).argmax().item(), out)
            # Resize the CAM and overlay it
            result = overlay_mask(to_pil_image(img), to_pil_image(activation_map[0].squeeze(0), mode='F'), alpha=0.5)
            # Display it
            plt.imshow(result); plt.axis('off'); plt.tight_layout(); plt.show()




if __name__ == '__main__':
    network = eval(args.model.replace('-', '_'))()
    network.cuda()
    saved_model_dir = os.path.join(args.save_dir, args.exp, args.model + '.pth')

    if os.path.exists(saved_model_dir):
        print('==> Start testing, current model name: ' + args.model)
        network.load_state_dict(single(saved_model_dir))
    else:
        print('==> No existing trained model!')
        exit(0)

    dataset_dir = os.path.join(args.data_dir, args.folder)

    test_dataset = SingleLoader(dataset_dir)
    test_loader = DataLoader(test_dataset,
                             batch_size=1,
                             num_workers=args.num_workers,
                             pin_memory=True)

    result_dir = os.path.join(args.result_dir, args.folder, args.model)
    test(test_loader, network, result_dir)
