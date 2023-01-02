from typing import List
from Dataset.CustomDataset import AgeGroupAndAgeDataset, StandardDataset, AgeDatasetKL
from Dataset.CustomDataLoaders import CustomDataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from Utils import AAR, CSVUtils, AgeConversion
from Utils.Validator import Validator

#Caricamento del dataframe
df = CSVUtils.get_df_from_csv("./training_caip_contest.csv", "./training_caip_contest/")

df_train, df_val = train_test_split(df, test_size=0.25, random_state=42)
df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)

from torchvision import transforms
import torch

transform_func = transforms.Compose([
    transforms.Resize(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.1, 0.1, 0.1, 0.1),
    transforms.RandAugment(2, 9),
    transforms.PILToTensor(),
    transforms.ConvertImageDtype(torch.float),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

transform_func_val = transforms.Compose([
    transforms.Resize(224),
    transforms.PILToTensor(),
    transforms.ConvertImageDtype(torch.float),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

# Implementazione di un Dataset utilizzando "CustomDataset" per l'architettura con Film
cd_train = AgeDatasetKL(df_train, path_col="path", label_col="age", label_function="Linear", 
                        transform_func=transform_func)
cd_train.set_n_classes(81)
cd_train.set_starting_class(1)
dm_train = CustomDataLoader(cd_train)
dl_train = dm_train.get_unbalanced_dataloader(batch_size=64, shuffle=True, drop_last=True, num_workers=12, prefetch_factor=4, pin_memory=True)

cd_val = StandardDataset(df_val, path_col="path", label_col="age", label_function="CAE", transform_func=transform_func_val)
cd_val.set_n_classes(81)
cd_val.set_starting_class(1)
validator = Validator(cd_val, AgeConversion.EVAge, 32, num_workers=8, prefetch_factor=4)

from ResNetFilmed.resnet import ResNetFiLMed, BackBone, ResNetNotFiLMed, DoNothingLayer
from torchvision.models import resnet18, ResNet18_Weights, efficientnet_b0, EfficientNet_B0_Weights
import torch
from torch import optim
import torch.nn.functional as F
from torch import nn

####################################################
EPOCHS = 12
####################################################

backbone = resnet18(ResNet18_Weights.IMAGENET1K_V1)
backbone.fc = DoNothingLayer()
backbone.train()
backbone.requires_grad_(True)
backbone.to("cuda")
model_age = ResNetNotFiLMed(backbone, 81)
opt = optim.SGD(set([*backbone.parameters(), *model_age.fc0.parameters()]), lr=0.1, weight_decay=5e-4)
scheduler = optim.lr_scheduler.OneCycleLR(opt, max_lr=0.1, steps_per_epoch=len(dl_train), epochs=EPOCHS, three_phase=True)
kl = nn.KLDivLoss(reduction="batchmean")

best_val_aar = -1

for e in range(EPOCHS):
    with tqdm(dl_train, unit=" batch") as tepoch:
        for batch in tepoch:
            opt.zero_grad()
            x, y = batch
            x = x.to("cuda")
            y_age: torch.Tensor = y[0].to("cuda")
            y_age_kl: torch.Tensor = y[1].to("cuda")

            out_age = model_age(x)
            loss_age_kl: torch.Tensor = kl(F.log_softmax(out_age, dim=-1), y_age_kl)

            out = F.softmax(out_age, dim=-1)
            out = AgeConversion.EVAge(out).to("cuda")
            loss_age = torch.mean(torch.abs(y_age - out))

            loss = loss_age_kl + loss_age
            loss.backward()
            opt.step()
            scheduler.step()

            tepoch.set_postfix(loss_age_kl=loss_age_kl.detach().cpu().numpy(), loss_age=loss_age.detach().cpu().numpy())

    def forward_function(x):
        out = model_age(x)
        out = F.softmax(out, dim=-1)
        return out

    val_aar, val_aar_old = validator.validate(forward_function)
    print(val_aar, val_aar_old)

    if val_aar > best_val_aar:
        best_val_aar = val_aar
        torch.save(model_age.state_dict(), "./model_age_feature_simple.pt")
        print("Saved model")

####################################################
EPOCHS = 8
####################################################

dl_train = dm_train.get_balanced_class_dataloader(class_ranges=[(0, 11), (11, 21), (21, 31), (31, 41), (41, 51), (51, 61), (61, 71), (71, 91)], 
                                                  batch_size=64, num_workers=12, prefetch_factor=4, pin_memory=True)

model_age.load_state_dict(torch.load("./model_age_feature_simple.pt"))
opt = optim.SGD(set([*model_age.fc0.parameters()]), lr=0.1, weight_decay=5e-4)
scheduler = optim.lr_scheduler.OneCycleLR(opt, max_lr=0.1, steps_per_epoch=len(dl_train), epochs=EPOCHS, three_phase=True)

def forward_function(x):
    out = model_age(x)
    out = F.softmax(out, dim=-1)
    return out

mae, val_aar, val_aar_old = validator.validate_ext(forward_function)
print(mae, val_aar, val_aar_old)

best_val_aar = val_aar

for e in range(EPOCHS):
    with tqdm(dl_train, unit=" batch") as tepoch:
        for batch in tepoch:
            opt.zero_grad()
            x, y = batch
            x = x.to("cuda")
            y_age: torch.Tensor = y[0].to("cuda")
            y_age_kl: torch.Tensor = y[1].to("cuda")

            out_age = model_age(x)
            loss_age_kl: torch.Tensor = kl(F.log_softmax(out_age, dim=-1), y_age_kl)

            out = F.softmax(out_age, dim=-1)
            out = AgeConversion.EVAge(out).to("cuda")
            loss_age = torch.mean(torch.abs(y_age - out))

            loss = loss_age_kl + torch.square(loss_age - mae)
            loss.backward()
            opt.step()
            scheduler.step()

            tepoch.set_postfix(loss_age_kl=loss_age_kl.detach().cpu().numpy(), loss_age=loss_age.detach().cpu().numpy())

    def forward_function(x):
        out = model_age(x)
        out = F.softmax(out, dim=-1)
        return out

    val_aar, val_aar_old = validator.validate(forward_function)
    print(val_aar, val_aar_old)

    if val_aar > best_val_aar:
        best_val_aar = val_aar
        torch.save(model_age.state_dict(), "./model_age_classification_simple.pt")
        print("Saved model")