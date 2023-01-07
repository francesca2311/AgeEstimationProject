from Dataset.CustomDataLoaders import CustomDataLoader
from Dataset import CustomDataset
from tqdm import tqdm
import torch
from torch.nn import functional as F


class Validator:
    def __init__(self, custom_dataset: CustomDataset, label_function, batch_size: int=32, **kwargs) -> None:
        dm_val: CustomDataLoader = CustomDataLoader(custom_dataset)
        self.dl_val = dm_val.get_unbalanced_dataloader(batch_size=batch_size, shuffle=False, **kwargs)
        self.label_function = label_function
    
    def validate(self, forward_function):
        ae_for_age_group = {x: [] for x in range(8)}

        errors = []
        with tqdm(self.dl_val, unit=" batch") as tepoch:
            for batch in tepoch:
                with torch.no_grad():
                    x, y = batch
                    x = x.to("cuda")
                    y = y.to("cuda")

                    out = forward_function(x)

                    for y_real, y_pred in zip(self.label_function(y), self.label_function(out)):
                        y_pred = torch.round(y_pred)
                        ae = torch.abs(y_real - y_pred)
                        errors.append(ae)

                        idx = 7 if int((y_real)/10) > 7 else int((y_real)/10)

                        ae_for_age_group[idx] += [ae]
        # MAE
        mae = torch.mean(torch.tensor(errors))
        
        # MMAE
        mmae = 0
        for k in ae_for_age_group:
            ae_for_age_group[k] = torch.mean(torch.tensor(ae_for_age_group[k]))
            mmae += ae_for_age_group[k]
        mmae = mmae / len(ae_for_age_group)

        # SIGMA
        sigma = 0
        for k in ae_for_age_group:
            sigma += torch.square(ae_for_age_group[k] - mae)
        sigma = sigma / len(ae_for_age_group)
        sigma = torch.sqrt(sigma)
        
        return (torch.max(torch.tensor(0), 5 - mmae) + torch.max(torch.tensor(0), 5 - sigma), torch.max(torch.tensor(0), 7 - mae) + torch.max(torch.tensor(0), 3 - sigma))

    def validate_ext(self, forward_function):
        ae_for_age_group = {x: [] for x in range(8)}

        errors = []
        with tqdm(self.dl_val, unit=" batch") as tepoch:
            for batch in tepoch:
                with torch.no_grad():
                    x, y = batch
                    x = x.to("cuda")
                    y = y.to("cuda")

                    out = forward_function(x)

                    for y_real, y_pred in zip(self.label_function(y), self.label_function(out)):
                        y_pred = torch.round(y_pred)
                        ae = torch.abs(y_real - y_pred)
                        errors.append(ae)

                        idx = 7 if int((y_real)/10) > 7 else int((y_real)/10)

                        ae_for_age_group[idx] += [ae]
        # MAE
        mae = torch.mean(torch.tensor(errors))
        
        # MMAE
        mmae = 0
        for k in ae_for_age_group:
            ae_for_age_group[k] = torch.mean(torch.tensor(ae_for_age_group[k]))
            mmae += ae_for_age_group[k]
        mmae = mmae / len(ae_for_age_group)

        # SIGMA
        sigma = 0
        for k in ae_for_age_group:
            sigma += torch.square(ae_for_age_group[k] - mae)
        sigma = sigma / len(ae_for_age_group)
        sigma = torch.sqrt(sigma)
        
        return (mae, torch.max(torch.tensor(0), 5 - mmae) + torch.max(torch.tensor(0), 5 - sigma), torch.max(torch.tensor(0), 7 - mae) + torch.max(torch.tensor(0), 3 - sigma))

    def validate_ext_fixed(self, forward_function):
        ae_for_age_group = {x: [] for x in range(8)}

        errors = []
        with tqdm(self.dl_val, unit=" batch") as tepoch:
            for batch in tepoch:
                with torch.no_grad():
                    x, y = batch
                    x = x.to("cuda")
                    y = y.to("cuda")

                    out = forward_function(x)

                    for y_real, y_pred in zip(self.label_function(y), self.label_function(out)):
                        y_pred = torch.round(y_pred)
                        ae = torch.abs(y_real - y_pred)
                        errors.append(ae)

                        idx = 7 if int((y_real)/10) > 7 else int((y_real)/10)

                        ae_for_age_group[idx] += [ae]
        # MAE
        mae = torch.mean(torch.tensor(errors))
        
        # MMAE
        mmae = 0
        for k in ae_for_age_group:
            ae_for_age_group[k] = torch.mean(torch.tensor(ae_for_age_group[k]))
            mmae += ae_for_age_group[k]
        mmae = mmae / len(ae_for_age_group)

        # SIGMA
        sigma = 0
        for k in ae_for_age_group:
            sigma += torch.square(ae_for_age_group[k] - mae)
        sigma = sigma / len(ae_for_age_group)
        sigma = torch.sqrt(sigma)
        
        return (mae, torch.max(torch.tensor(0), 5 - mmae) + torch.max(torch.tensor(0), 5 - sigma), torch.max(torch.tensor(0), 7 - mae) + torch.max(torch.tensor(0), 3 - sigma))

    def validate_ext2(self, forward_function):
        ae_for_age_group = {x: [] for x in range(8)}

        errors = []
        with tqdm(self.dl_val, unit=" batch") as tepoch:
            for batch in tepoch:
                with torch.no_grad():
                    x, y = batch
                    x = x.to("cuda")
                    y = y.to("cuda")

                    out = forward_function(x)

                    for y_real, y_pred in zip(self.label_function(y), self.label_function(out)):
                        y_pred = torch.round(y_pred)
                        ae = torch.abs(y_real - y_pred)
                        errors.append(ae)

                        idx = 7 if int((y_real)/10) > 7 else int((y_real)/10)

                        ae_for_age_group[idx] += [ae]
        # MAE
        mae = torch.mean(torch.tensor(errors))
        
        # MMAE
        mmae = 0
        for k in ae_for_age_group:
            ae_for_age_group[k] = torch.mean(torch.tensor(ae_for_age_group[k]))
            mmae += ae_for_age_group[k]
        mmae = mmae / len(ae_for_age_group)

        # SIGMA
        sigma = 0
        for k in ae_for_age_group:
            sigma += torch.square(ae_for_age_group[k] - mae)
        sigma = sigma / len(ae_for_age_group)
        sigma = torch.sqrt(sigma)
        
        return (ae_for_age_group, mae, torch.max(torch.tensor(0), 5 - mmae) + torch.max(torch.tensor(0), 5 - sigma), torch.max(torch.tensor(0), 7 - mae) + torch.max(torch.tensor(0), 3 - sigma))

    def validate_ext3(self, forward_function):
        ae_for_age_group = {x: [] for x in range(8)}
        ae_for_age = {x: [] for x in range(81)}

        errors = []
        with tqdm(self.dl_val, unit=" batch") as tepoch:
            for batch in tepoch:
                with torch.no_grad():
                    x, y = batch
                    x = x.to("cuda")
                    y = y.to("cuda")

                    out = forward_function(x)

                    for y_real, y_pred in zip(self.label_function(y), self.label_function(out)):
                        y_pred = torch.round(y_pred)
                        ae = torch.abs(y_real - y_pred)
                        errors.append(ae)

                        idx = 7 if int((y_real)/10) > 7 else int((y_real)/10)

                        ae_for_age_group[idx] += [ae]
                        ae_for_age[int(y_real)] += [ae]
        # MAE
        mae = torch.mean(torch.tensor(errors))
        
        # MMAE
        mmae = 0
        for k in ae_for_age_group:
            ae_for_age_group[k] = torch.mean(torch.tensor(ae_for_age_group[k]))
            mmae += ae_for_age_group[k]
        mmae = mmae / len(ae_for_age_group)

        for k in ae_for_age:
            ae_for_age[k] = torch.mean(torch.tensor(ae_for_age[k]))

        # SIGMA
        sigma = 0
        for k in ae_for_age_group:
            sigma += torch.square(ae_for_age_group[k] - mae)
        sigma = sigma / len(ae_for_age_group)
        sigma = torch.sqrt(sigma)
        
        return (ae_for_age, ae_for_age_group, mae, torch.max(torch.tensor(0), 5 - mmae) + torch.max(torch.tensor(0), 5 - sigma), torch.max(torch.tensor(0), 7 - mae) + torch.max(torch.tensor(0), 3 - sigma))