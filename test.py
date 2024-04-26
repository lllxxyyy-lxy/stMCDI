import argparse
import torch
import datetime
import json
import yaml
import os
import warnings
from src.main_model_st import stMCDI
from src.utils_st import train, evaluate
from dataset import get_dataloader

warnings.filterwarnings("ignore", category=DeprecationWarning)
parser = argparse.ArgumentParser(description="stMCDI")
parser.add_argument("--config", type=str, default="st_config.yaml")
parser.add_argument("--device", default="cuda", help="Device")
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--testmissingratio", type=float, default=0.2)
parser.add_argument("--nfold", type=int, default=5, help="for 5-fold test")
parser.add_argument("--unconditional", action="store_true", default=0)
parser.add_argument("--modelfolder", type=str, default="")
parser.add_argument("--nsample", type=int, default=100)

args = parser.parse_args()
print(args)

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

path = "/data/minwenwen/lixiaoyu/DiffImpute/config/" + args.config
with open(path, "r") as f:
    config = yaml.safe_load(f)

config["model"]["is_unconditional"] = args.unconditional
config["model"]["test_missing_ratio"] = args.testmissingratio

print(json.dumps(config, indent=4))

# Create folder
current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
foldername = "./save/st_fold" + str(args.nfold) + "_" + current_time + "/"
print("model folder:", foldername)
os.makedirs(foldername, exist_ok=True)
with open(foldername + "config.json", "w") as f:
    json.dump(config, f, indent=4)

# Every loader contains "observed_data", "observed_mask", "gt_mask", "timepoints"
train_loader, valid_loader, test_loader = get_dataloader(
    seed=args.seed,
    nfold=args.nfold,
    batch_size=config["train"]["batch_size"],
    missing_ratio=config["model"]["test_missing_ratio"],
)

model = stMCDI(config, args.device).to(args.device)
# print("model.target_dim:", model.target_dim)
model.load_state_dict(torch.load('/data/minwenwen/lixiaoyu/DiffImpute/save/st_fold5_20230920_095933/model.pth'))
total_params = sum(p.numel() for p in model.parameters())

print(model)
print("Total number of parameters:", total_params)