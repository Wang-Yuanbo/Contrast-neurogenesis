import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import torch.nn.init as init
import torch.multiprocessing as mp

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

import datetime
import os
import copy


def load_dataset(which_mnist, batch_size):    
    transform = transforms.Compose([transforms.ToTensor()])
    
    if which_mnist == 'o':
        mnist_trainset_full = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        mnist_testset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    
    if which_mnist == 'f':
        mnist_trainset_full = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
        mnist_testset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
    
    if which_mnist == 'k':
        mnist_trainset_full = datasets.KMNIST(root='./data', train=True, download=True, transform=transform)
        mnist_testset = datasets.KMNIST(root='./data', train=False, download=True, transform=transform)
    
    if which_mnist == 'e':
        mnist_trainset_full = datasets.EMNIST(root='./data', split='balanced', train=True, download=True, transform=transform)
        mnist_testset = datasets.EMNIST(root='./data', split='balanced', train=False, download=True, transform=transform)
    
    if which_mnist == 'q':
        mnist_trainset_full = datasets.QMNIST(root='./data', train=True, download=True, transform=transform)
        mnist_testset = datasets.QMNIST(root='./data', train=False, download=True, transform=transform)
    
    train_size = int(0.9 * len(mnist_trainset_full))  
    val_size = len(mnist_trainset_full) - train_size  
    mnist_trainset, mnist_valset = random_split(mnist_trainset_full, [train_size, val_size])
    
    train_loader = DataLoader(mnist_trainset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(mnist_valset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(mnist_testset, batch_size=batch_size, shuffle=True)
    
    input_size, output_size = get_input_output_sizes(mnist_testset)
    
    return train_loader, val_loader, test_loader, input_size, output_size


def file_print(*args, sep=' ', end='\n', flush=False, folder_path):

    print_output_file_name = f'{folder_path}/print.log'
    os.makedirs(folder_path, exist_ok=True)

    output = sep.join(map(str, args))

    print(output)

    with open(print_output_file_name, 'a') as file:
        file.write(output + end)
        if flush:
            file.flush()


def get_input_output_sizes(dataset):    
    sample_data, sample_target = dataset[0]
    input_size = torch.prod(torch.tensor(sample_data.size()))  # (C, H, W)
    num_classes = len(dataset.classes)
    return input_size, num_classes

def layer_init(linear_layer):
    init.kaiming_uniform_(linear_layer.weight, nonlinearity='relu')
    if linear_layer.bias is not None:
        init.zeros_(linear_layer.bias)
    return linear_layer

def compute_cross_entropy(p, q):
    q = F.log_softmax(q, dim=-1)
    loss = torch.sum(p * q, dim=-1)
    return - loss.mean()

def stablize_logits(logits):
    logits_max, _ = torch.max(logits, dim=-1, keepdim=True)
    logits = logits - logits_max.detach()
    return logits

@torch.no_grad()
def concat_all_gather(tensor):
    tensors_gather = [torch.ones_like(tensor)
                      for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output


class SupConLoss(nn.Module):

    def __init__(self, device, temperature=0.1):
        super(SupConLoss, self).__init__()
        self.device = device
        self.temperature = temperature
        self.logits_mask = None

    def set_temperature(self, temp=0.1):
        self.temperature = temp

    def forward(self, all_feats, all_labels, contra_range=-1):
        all_feats = F.normalize(all_feats, dim=-1, p=2)
        
        if contra_range == -1:
            feats = all_feats
            labels = all_labels
        else:
            feats = all_feats[:contra_range,:]
            labels = all_labels[:contra_range]

        mask = torch.eq(labels.view(-1, 1),
                        all_labels.contiguous().view(1, -1)).float().to(self.device)
        self.logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(mask.shape[0]).view(-1, 1).to(self.device),
            0
        )

        mask = mask * self.logits_mask

        logits = torch.matmul(feats, all_feats.T) / self.temperature
        logits = logits - (1 - self.logits_mask) * 1e9

        logits = stablize_logits(logits)

        p = mask / mask.sum(1, keepdim=True).clamp(min=1.0)
        loss = compute_cross_entropy(p, logits)

        return loss


class SetFFN(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super(SetFFN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        self.hidden_dims = hidden_dims
        self.input_state = None
        self.hidden_state = None
        
        self.layers = nn.ModuleList()
        
        last_count = input_dim
        for count in self.hidden_dims:
            if count>0:
                self.layers.append(layer_init(nn.Linear(last_count, count)))
                last_count = count
        self.layers.append(layer_init(nn.Linear(last_count, output_dim)))

    def forward(self, x):
        self.input_state = copy.deepcopy(x)
        self.hidden_state = torch.empty(x.shape[0], 0).to(x.device)
        flag = False
        for layer in self.layers:
            if flag:
                self.hidden_state = torch.cat((self.hidden_state, x), dim=1)
            flag = True
            x = F.relu(layer(x))
        return x


class WithHead(nn.Module):
    def __init__(self, feature_encoder, with_input_neuron, with_hidden_neuron, with_output_neuron, gradient_osmosis, deepcopy, feature_dim=128, temperature=0.1):
        super(WithHead, self).__init__()
        if deepcopy:
            self.feature_encoder = copy.deepcopy(feature_encoder)
        else:
            self.feature_encoder = feature_encoder
        self.with_input_neuron = with_input_neuron
        self.with_hidden_neuron = with_hidden_neuron
        self.with_output_neuron = with_output_neuron
        self.gradient_osmosis = gradient_osmosis
        
        input_dim = 0
        if with_input_neuron:
            input_dim += feature_encoder.input_dim
        if with_hidden_neuron:
            input_dim += sum(feature_encoder.hidden_dims)
        if with_output_neuron:
            input_dim += feature_encoder.output_dim
        self.contra_encoder = layer_init(nn.Linear(input_dim, feature_dim))
        
        
    def forward(self, x):
        if self.gradient_osmosis:
            y = self.feature_encoder(x)
        else:
            with torch.no_grad(): 
                y = self.feature_encoder(x)
        
        state = torch.empty(y.shape[0], 0).to(y.device)
        if self.with_input_neuron:
            state = torch.cat((state, self.feature_encoder.input_state), dim=1)
        if self.with_hidden_neuron:
            state = torch.cat((state, self.feature_encoder.hidden_state), dim=1)
        if self.with_output_neuron:
            state = torch.cat((state, y), dim=1)
                
        return self.contra_encoder(state)


def train(train_epoch, loader, train_model, criterion, train_optimizer, scheduler, neuron_counts, writer, is_contra, is_report, device, is_mocogrow=False):
    train_model.train()
    correct = 0
    total = 0
    accuracy = None
    loss = None
    
    for batch_idx, (data, target) in enumerate(loader):
        data = data.view(data.shape[0], -1)
        data, target = data.to(device), target.to(device)
    
        train_optimizer.zero_grad()
        output = train_model(data)
        loss = criterion(output, target)
        if is_mocogrow:
            loss = train_model.contra(data, target) + loss
        loss.backward()
        
        train_optimizer.step()
        
        current_lr = train_optimizer.param_groups[0]['lr']
        scheduler.step()
        
        if is_contra:
            if is_report:
                if batch_idx % 20 == 0:
                    print(f'contra neuron_count:{neuron_counts} Epoch: {train_epoch} [{batch_idx * len(data)}/{len(loader) * len(data)} '
                          f'({100. * batch_idx / len(loader):.0f}%)]\tLoss: {loss.item():.6f} '
                          f'LR: {current_lr:.6f}')
                    writer.add_scalar(f'contra_training_loss', loss.item(), train_epoch * len(loader) + batch_idx)
            
        else:
            _, predicted = torch.max(output.data, dim=1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            accuracy = 100 * correct / total
            
            if is_report:
                if batch_idx % 20 == 0:
                    print(f'predict neuron_count:{neuron_counts} Epoch: {train_epoch} [{batch_idx * len(data)}/{len(loader) * len(data)} '
                          f'({100. * batch_idx / len(loader):.0f}%)]\tLoss: {loss.item():.6f} '
                          f'Accuracy: {accuracy:.2f}% LR: {current_lr:.6f}')
    
                    writer.add_scalar(f'training_loss', loss.item(), train_epoch * len(loader) + batch_idx)
                    writer.add_scalar(f'training_accuracy', accuracy, train_epoch * len(loader) + batch_idx)  
                    
    if is_contra:
        return -1, loss.item()
    else:
        return accuracy, loss.item()

def validate(epoch, val_model, loader, criterion, val_or_test, writer, is_contra, is_report, device):
    val_model.eval()  
    val_loss = 0
    correct = 0
    total = 0
    accuracy = 0
    with torch.no_grad():  
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            data = data.view(data.shape[0], -1)  

            output = val_model(data)
            loss = criterion(output, target)
            val_loss += loss.item()  
            
            if not is_contra:
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
                accuracy += 100 * correct / total
    
    val_loss /= len(loader) 
    if is_contra:
        if is_report:
            if val_or_test == 'val':
                writer.add_scalar(f'contra_validate_loss', val_loss, epoch * len(loader))
                print(f'-----------------'
                      f'validation \tLoss: {val_loss:.6f} ----------------')
            elif val_or_test == 'test':
                print(f'-----------------'
                      f'test \tLoss: {val_loss:.6f} ----------------')
        return -1, val_loss
        
    else:
        accuracy /= len(loader)
        if is_report:
            if val_or_test == 'val':
                writer.add_scalar(f'validate_loss', val_loss, epoch * len(loader))
                writer.add_scalar(f'validate_accuracy', accuracy, epoch * len(loader))
                print(f'-----------------'
                      f'validation \tLoss: {val_loss:.6f} '
                      f'Accuracy: {accuracy:.2f}% ----------------')
            elif val_or_test == 'test':
                print(f'-----------------'
                      f'test \tLoss: {val_loss:.6f} '
                      f'Accuracy: {accuracy:.2f}% ----------------')
        return accuracy, val_loss

def save_heatmap(array, output_dir, filename, figsize=(10,8), current_row=-1):
    
    shape = list(array.shape)
    if current_row != -1:
        shape[0] = current_row+1
    df = pd.DataFrame(array[:shape[0],:], columns=[f'l2_{i}' for i in range(shape[1])], index=[f'l1_{i+1}' for i in range(shape[0])])
    
    os.makedirs(output_dir, exist_ok=True)

    csv_path = os.path.join(output_dir, f"{filename}.csv")
    df.to_csv(csv_path, index=True) 

    plt.figure(figsize=figsize)
    sns.heatmap(df, annot=False, fmt=".2f", cmap="viridis")

    plt.title(filename)

    heatmap_path = os.path.join(output_dir, f"{filename}.png")
    plt.savefig(heatmap_path)
    plt.close()

def learning(model, 
             is_contra,
             is_mocogrow,
             lr, 
             epoch, 
             hidden_config, 
             train_loader,
             test_loader,
             val_loader,
             train_writer, 
             sum_writer, 
             is_report,
             train_loss_note, 
             val_loss_note, 
             test_loss_note, 
             device,
             train_ac_note=None, 
             val_ac_note=None, 
             test_ac_note=None,
             ):
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    if is_mocogrow:
        model.get_optimizer(op=optimizer)

    if is_contra:
        criterion = SupConLoss(device=device, temperature=0.1).to(device)
    else:
        criterion = nn.CrossEntropyLoss()
    
    def lambda_lr(counter, epoch=epoch, train_loader_len=len(train_loader)):
        denominator = epoch * train_loader_len
        return 1 - counter / denominator

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_lr)

    last_train_accuracy=None
    last_train_loss=None
    last_val_accuracy=None
    last_val_loss=None
    for epoch in range(epoch):
        last_train_accuracy, last_train_loss = train(
              train_epoch=epoch, 
              loader=train_loader, 
              train_model=model,
              device=device,
              criterion=criterion, 
              train_optimizer=optimizer,
              scheduler=scheduler, 
              neuron_counts=hidden_config, 
              writer=train_writer,
              is_contra=is_contra,
              is_report=is_report,
              is_mocogrow=is_mocogrow)

        last_val_accuracy, last_val_loss = validate(
                 epoch=epoch,
                 val_model=model, 
                 loader=val_loader,
                 device=device,
                 criterion=criterion, 
                 val_or_test='val',
                 writer=train_writer,
                 is_contra=is_contra,
                 is_report=is_report)
    last_test_accuracy, last_test_loss = validate(
             epoch=epoch,
             val_model=model, 
             loader=test_loader, 
             device=device,
             criterion=criterion, 
             val_or_test='test',
             writer=sum_writer,
             is_contra=is_contra,
             is_report=is_report)

    train_loss_note[hidden_config[0]-1, hidden_config[1]]  = last_train_loss
    val_loss_note[hidden_config[0]-1, hidden_config[1]]    = last_val_loss
    test_loss_note[hidden_config[0]-1, hidden_config[1]]   = last_test_loss  
    
    if not is_contra:
        train_ac_note[hidden_config[0]-1, hidden_config[1]]    = last_train_accuracy
        val_ac_note[hidden_config[0]-1, hidden_config[1]]      = last_val_accuracy
        test_ac_note[hidden_config[0]-1, hidden_config[1]]     = last_test_accuracy


def detect_losses_in2d(model_class, 
                       hidden_config_2d_from,
                       hidden_config_2d_end,
                       input_dim, 
                       output_dim, 
                       train_loader, 
                       val_loader, 
                       test_loader,
                       normal_train,
                       train_lr, 
                       train_epoch,
                       contra_head, 
                       contra_lr, 
                       contra_epoch, 
                       predict_head,
                       pre_head_lr, 
                       pre_head_epoch,
                       model_save,
                       heatmaps_refresh_cycle,
                       contra_input,
                       contra_hidden,
                       contra_output, 
                       gradient_osmosis,
                       is_report,
                       device,
                       folder_path,):
    
    file_print(f'model_class={model_class}\n', 
               f'hidden_config_2d_from={hidden_config_2d_from}\n', 
               f'hidden_config_2d_end={hidden_config_2d_end}\n', 
               f'input_dim={input_dim}\n', 
               f'output_dim={output_dim}\n', 
               f'train_loader={train_loader}\n', 
               f'val_loader={val_loader}\n', 
               f'test_loader={test_loader}\n', 
               f'normal_train={normal_train}\n',
               f'train_lr={train_lr}\n', 
               f'train_epoch={train_epoch}\n',
               f'contra_head={contra_head}\n', 
               f'contra_lr={contra_lr}\n', 
               f'contra_epoch={contra_epoch}\n',
               f'predict_head={predict_head}\n',
               f'pre_head_lr={pre_head_lr}\n',
               f'pre_head_epoch={pre_head_epoch}\n',
               f'model_save={model_save}\n',
               f'heatmaps_refresh_cycle={heatmaps_refresh_cycle}\n',
               f'contra_input={contra_input}\n',
               f'contra_hidden={contra_hidden}\n',
               f'contra_output={contra_output}\n',
               f'gradient_osmosis={gradient_osmosis}\n',
               f'is_report={is_report}\n',
               f'device={device}\n',
               f'folder_path={folder_path}\n',
               folder_path=folder_path)
    
    torch.autograd.set_detect_anomaly(False)
    os.makedirs(folder_path, exist_ok=True)
    if is_report:
        sum_writer = SummaryWriter(f'{folder_path}/runs/tensorboard/sum/predict')
    else:
        sum_writer = None
    
    note_size = [hidden_config_2d_end[0], hidden_config_2d_end[1]+1]
    
    if normal_train:
        predict_train_ac_note = np.zeros(note_size)-1
        predict_train_loss_note = np.zeros(note_size)-1
        predict_val_ac_note = np.zeros(note_size)-1
        predict_val_loss_note = np.zeros(note_size)-1
        predict_test_ac_note = np.zeros(note_size)-1
        predict_test_loss_note = np.zeros(note_size)-1
    
    if contra_head:
        contra_train_loss_note = np.zeros(note_size)-1
        contra_val_loss_note = np.zeros(note_size)-1
        contra_test_loss_note = np.zeros(note_size)-1    
    
    if predict_head:
        pre_head_train_ac_note = np.zeros(note_size)-1
        pre_head_train_loss_note = np.zeros(note_size)-1
        pre_head_val_ac_note = np.zeros(note_size)-1
        pre_head_val_loss_note = np.zeros(note_size)-1
        pre_head_test_ac_note = np.zeros(note_size)-1
        pre_head_test_loss_note = np.zeros(note_size)-1
    

    for d1 in range(hidden_config_2d_from[0], hidden_config_2d_end[0]):
        for d2 in range(hidden_config_2d_from[1]+1, hidden_config_2d_end[1]+1):
            hidden_config = [d1+1,d2]
            
            if is_report:
                train_writer = SummaryWriter(f'{folder_path}/runs/tensorboard/train/neuron_counts={hidden_config}')
            else:
                train_writer = None
                
            model = model_class(input_dim, hidden_config , output_dim).to(device)
            
            if normal_train:
                learning(model=model, 
                         is_contra=False, 
                         lr=train_lr,
                         epoch=train_epoch, 
                         hidden_config=hidden_config, 
                         train_loader=train_loader,
                         test_loader=test_loader,
                         val_loader=val_loader,
                         train_writer=train_writer, 
                         sum_writer=sum_writer, 
                         is_report=is_report,
                         train_loss_note=predict_train_loss_note, 
                         val_loss_note=predict_val_loss_note, 
                         test_loss_note=predict_test_loss_note, 
                         train_ac_note=predict_train_ac_note, 
                         val_ac_note=predict_val_ac_note, 
                         test_ac_note=predict_test_ac_note,
                         device=device,
                         is_mocogrow=False)
            
            if model_save:
                os.makedirs(f'{folder_path}/models/', exist_ok=True)    # 创建文件夹
                torch.save(model.state_dict(), f'{folder_path}/models/neuron_count={hidden_config}') 

            if contra_head:
                contra_criterion = SupConLoss(device=device, temperature=0.1).to(device)
                
                model_with_contra_head = WithHead(model,
                                                  with_input_neuron=contra_input,
                                                  with_hidden_neuron=contra_hidden,
                                                  with_output_neuron=contra_output,
                                                  gradient_osmosis=gradient_osmosis, 
                                                  deepcopy=True, 
                                                  feature_dim=32).to(device)
                learning(model=model_with_contra_head, 
                         is_contra=True, 
                         lr=contra_lr,
                         epoch=contra_epoch, 
                         hidden_config=hidden_config, 
                         train_loader=train_loader,
                         test_loader=test_loader,
                         val_loader=val_loader,
                         train_writer=train_writer, 
                         sum_writer=sum_writer, 
                         is_report=is_report,
                         train_loss_note=contra_train_loss_note, 
                         val_loss_note=contra_val_loss_note, 
                         test_loss_note=contra_test_loss_note,
                         device=device,
                         is_mocogrow=False)
                
                if model_save:
                    os.makedirs(f'{folder_path}/model/with_contra_head/', exist_ok=True)    # 创建文件夹
                    torch.save(model.state_dict(), f'{folder_path}/model/with_contra_head/neuron_count={hidden_config}')
                
            if predict_head:
                model_with_pre_head = WithHead(model,
                                               with_input_neuron=contra_input,
                                               with_hidden_neuron=contra_hidden,
                                               with_output_neuron=contra_output,
                                               gradient_osmosis=gradient_osmosis, 
                                               deepcopy=True,
                                               feature_dim=output_dim).to(device)
                learning(model=model_with_pre_head, 
                         is_contra=False, 
                         lr=pre_head_lr,
                         epoch=pre_head_epoch, 
                         hidden_config=hidden_config, 
                         train_loader=train_loader,
                         test_loader=test_loader,
                         val_loader=val_loader,
                         train_writer=train_writer, 
                         sum_writer=sum_writer, 
                         is_report=is_report,
                         train_loss_note=pre_head_train_loss_note, 
                         val_loss_note=pre_head_val_loss_note, 
                         test_loss_note=pre_head_test_loss_note, 
                         train_ac_note=pre_head_train_ac_note, 
                         val_ac_note=pre_head_val_ac_note, 
                         test_ac_note=pre_head_test_ac_note,
                         device=device,
                         is_mocogrow=False)
                
                if model_save:
                    os.makedirs(f'{folder_path}/models/with_predict_head/', exist_ok=True)    # 创建文件夹
                    torch.save(model_with_pre_head.state_dict(), f'{folder_path}/models/with_predict_head/neuron_count={hidden_config}') 
                    
            if is_report:
                train_writer.close()
            
            
        if d1 %heatmaps_refresh_cycle == 0 or d1 == hidden_config_2d_end[0]-1:
            if normal_train:
                save_heatmap(predict_train_ac_note, output_dir=f'{folder_path}/heatmaps/predict', filename='Predict Train Accuracy', current_row=d1)
                save_heatmap(predict_train_loss_note, output_dir=f'{folder_path}/heatmaps/predict', filename='Predict Train Loss', current_row=d1)
                save_heatmap(predict_val_ac_note, output_dir=f'{folder_path}/heatmaps/predict', filename='Predict Val Accuracy', current_row=d1)
                save_heatmap(predict_val_loss_note, output_dir=f'{folder_path}/heatmaps/predict', filename='Predict Val Loss', current_row=d1)
                save_heatmap(predict_test_ac_note, output_dir=f'{folder_path}/heatmaps/predict', filename='Predict Test Accuracy', current_row=d1)
                save_heatmap(predict_test_loss_note, output_dir=f'{folder_path}/heatmaps/predict', filename='Predict Test Loss', current_row=d1)
            
            if contra_head:
                save_heatmap(contra_train_loss_note, output_dir=f'{folder_path}/heatmaps/contra', filename='Contrastive Train Loss', current_row=d1)
                save_heatmap(contra_val_loss_note, output_dir=f'{folder_path}/heatmaps/contra', filename='Contrastive Val Loss', current_row=d1)
                save_heatmap(contra_test_loss_note, output_dir=f'{folder_path}/heatmaps/contra', filename='Contrastive Test Loss', current_row=d1)
                
            if predict_head:
                save_heatmap(pre_head_train_ac_note, output_dir=f'{folder_path}/heatmaps/pre_head', filename='With Predict Head Train Accuracy', current_row=d1)
                save_heatmap(pre_head_train_loss_note, output_dir=f'{folder_path}/heatmaps/pre_head', filename='With Predict Head Train Loss', current_row=d1)
                save_heatmap(pre_head_val_ac_note, output_dir=f'{folder_path}/heatmaps/pre_head', filename='With Predict Head Val Accuracy', current_row=d1)
                save_heatmap(pre_head_val_loss_note, output_dir=f'{folder_path}/heatmaps/pre_head', filename='With Predict Head Val Loss', current_row=d1)
                save_heatmap(pre_head_test_ac_note, output_dir=f'{folder_path}/heatmaps/pre_head', filename='With Predict Head Test Accuracy', current_row=d1)
                save_heatmap(pre_head_test_loss_note, output_dir=f'{folder_path}/heatmaps/pre_head', filename='With Predict Head Test Loss', current_row=d1)
    
    if is_report:
        sum_writer.close()
    else:
        pass



def gpu_work_check_smooth(gpu_id,
                          which_mnist,
                        gradient_osmosis,
                        contra_hidden,
                        contra_input,
                        contra_output,
                        normal_train,
                        contra_head,
                        predict_head,
                        hidden_config_2d_from, 
                        hidden_config_2d_end,
                        heatmaps_refresh_cycle,
                        train_lr,
                        train_epoch,
                        contra_lr,
                        contra_epoch,
                        pre_head_lr,
                        pre_head_epoch,
                        model_save,
                        is_report,
                        batch_size):
    device = torch.device(f"cuda:{gpu_id}")
    
    folder_path = f'log/{hidden_config_2d_from}-{hidden_config_2d_end}_{which_mnist}_contrahi{contra_hidden}_gradosm{gradient_osmosis}_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")}'
    
    train_loader, val_loader, test_loader, input_size, output_size = load_dataset(which_mnist, batch_size)
    
    file_print(f"Input size: {input_size}",folder_path=folder_path)
    file_print(f"Number of classes: {output_size}",folder_path=folder_path)
    
    detect_losses_in2d(model_class=SetFFN, 
                       hidden_config_2d_from=hidden_config_2d_from, 
                       hidden_config_2d_end=hidden_config_2d_end,
                       input_dim=input_size, 
                       output_dim=output_size, 
                       train_loader=train_loader, 
                       val_loader=val_loader, 
                       test_loader=test_loader,
                       normal_train=normal_train,
                       train_lr=train_lr, 
                       train_epoch=train_epoch,
                       contra_head=contra_head,
                       contra_lr=contra_lr,
                       contra_epoch=contra_epoch,
                       predict_head=predict_head,
                       pre_head_lr=pre_head_lr,
                       pre_head_epoch=pre_head_epoch,
                       model_save=model_save,
                       heatmaps_refresh_cycle=heatmaps_refresh_cycle,
                       contra_input=contra_input,
                       contra_hidden=contra_hidden,
                       contra_output=contra_output, 
                       gradient_osmosis=gradient_osmosis,
                       is_report=is_report,
                       device=device,
                       folder_path=folder_path
                       )


def wrapped_run_function(rank):
    # param = {
    #     'which_mnist' : 'q',
    #     'gradient_osmosis':True,
    #     'contra_hidden':True,
    #     'contra_input':False,
    #     'contra_output':True,
    #     'normal_train':True,
    #     'contra_head':True,
    #     'predict_head':True,
    #     'hidden_config_2d_from':[1,2],
    #     'hidden_config_2d_end':[2,3],
    #     'heatmaps_refresh_cycle':1,
    #     'train_lr':0.004,
    #     'train_epoch':10,
    #     'contra_lr':0.004,
    #     'contra_epoch':10,
    #     'pre_head_lr':0.004,
    #     'pre_head_epoch':10,
    #     'model_save':False,
    #     'is_report':False,
    #     'batch_size':512
    #     }
    # 
    # options = [True, False]
    # # dateset_options = ['o', 'f', 'k', 'e', 'q']
    # dateset_options = ['f', 'k', 'e', 'q']
    param_list=[{
        'gpu_id':0,
        'which_mnist' : 'o',
        'gradient_osmosis':True,
        'contra_hidden':True,
        'contra_input':False,
        'contra_output':True,
        'normal_train':True,
        'contra_head':True,
        'predict_head':True,
        'hidden_config_2d_from':[34,-1],
        'hidden_config_2d_end':[60,30],
        'heatmaps_refresh_cycle':1,
        'train_lr':0.004,
        'train_epoch':10,
        'contra_lr':0.004,
        'contra_epoch':10,
        'pre_head_lr':0.004,
        'pre_head_epoch':10,
        'model_save':False,
        'is_report':False,
        'batch_size':512
        },{
        'gpu_id':1,
        'which_mnist' : 'f',
        'gradient_osmosis':True,
        'contra_hidden':True,
        'contra_input':False,
        'contra_output':True,
        'normal_train':True,
        'contra_head':True,
        'predict_head':True,
        'hidden_config_2d_from':[14,-1],
        'hidden_config_2d_end':[60,30],
        'heatmaps_refresh_cycle':1,
        'train_lr':0.004,
        'train_epoch':10,
        'contra_lr':0.004,
        'contra_epoch':10,
        'pre_head_lr':0.004,
        'pre_head_epoch':10,
        'model_save':False,
        'is_report':False,
        'batch_size':512
        },{
        'gpu_id':2,
        'which_mnist' : 'e',
        'gradient_osmosis':True,
        'contra_hidden':False,
        'contra_input':False,
        'contra_output':True,
        'normal_train':True,
        'contra_head':True,
        'predict_head':True,
        'hidden_config_2d_from':[21,-1],
        'hidden_config_2d_end':[60,30],
        'heatmaps_refresh_cycle':1,
        'train_lr':0.004,
        'train_epoch':10,
        'contra_lr':0.004,
        'contra_epoch':10,
        'pre_head_lr':0.004,
        'pre_head_epoch':10,
        'model_save':False,
        'is_report':False,
        'batch_size':512
        },{
        'gpu_id':3,
        'which_mnist' : 'e',
        'gradient_osmosis':True,
        'contra_hidden':True,
        'contra_input':False,
        'contra_output':True,
        'normal_train':True,
        'contra_head':True,
        'predict_head':True,
        'hidden_config_2d_from':[21,-1],
        'hidden_config_2d_end':[60,30],
        'heatmaps_refresh_cycle':1,
        'train_lr':0.004,
        'train_epoch':10,
        'contra_lr':0.004,
        'contra_epoch':10,
        'pre_head_lr':0.004,
        'pre_head_epoch':10,
        'model_save':False,
        'is_report':False,
        'batch_size':512
        },{
        'gpu_id':4,
        'which_mnist' : 'k',
        'gradient_osmosis':True,
        'contra_hidden':False,
        'contra_input':False,
        'contra_output':True,
        'normal_train':True,
        'contra_head':True,
        'predict_head':True,
        'hidden_config_2d_from':[33,-1],
        'hidden_config_2d_end':[60,30],
        'heatmaps_refresh_cycle':1,
        'train_lr':0.004,
        'train_epoch':10,
        'contra_lr':0.004,
        'contra_epoch':10,
        'pre_head_lr':0.004,
        'pre_head_epoch':10,
        'model_save':False,
        'is_report':False,
        'batch_size':512
        },{
        'gpu_id':5,
        'which_mnist' : 'k',
        'gradient_osmosis':True,
        'contra_hidden':True,
        'contra_input':False,
        'contra_output':True,
        'normal_train':True,
        'contra_head':True,
        'predict_head':True,
        'hidden_config_2d_from':[33,-1],
        'hidden_config_2d_end':[60,30],
        'heatmaps_refresh_cycle':1,
        'train_lr':0.004,
        'train_epoch':10,
        'contra_lr':0.004,
        'contra_epoch':10,
        'pre_head_lr':0.004,
        'pre_head_epoch':10,
        'model_save':False,
        'is_report':False,
        'batch_size':512
        },{
        'gpu_id':6,
        'which_mnist' : 'q',
        'gradient_osmosis':True,
        'contra_hidden':True,
        'contra_input':False,
        'contra_output':True,
        'normal_train':True,
        'contra_head':True,
        'predict_head':True,
        'hidden_config_2d_from':[36,-1],
        'hidden_config_2d_end':[60,30],
        'heatmaps_refresh_cycle':1,
        'train_lr':0.004,
        'train_epoch':10,
        'contra_lr':0.004,
        'contra_epoch':10,
        'pre_head_lr':0.004,
        'pre_head_epoch':10,
        'model_save':False,
        'is_report':False,
        'batch_size':512
        },{
        'gpu_id':7,
        'which_mnist' : 'q',
        'gradient_osmosis':True,
        'contra_hidden':False,
        'contra_input':False,
        'contra_output':True,
        'normal_train':True,
        'contra_head':True,
        'predict_head':True,
        'hidden_config_2d_from':[36,-1],
        'hidden_config_2d_end':[60,30],
        'heatmaps_refresh_cycle':1,
        'train_lr':0.004,
        'train_epoch':10,
        'contra_lr':0.004,
        'contra_epoch':10,
        'pre_head_lr':0.004,
        'pre_head_epoch':10,
        'model_save':False,
        'is_report':False,
        'batch_size':512
        }]
    
    try:
        gpu_work_check_smooth(**param_list[rank])
    except Exception as e:
        print(f"Error in process {rank}: {e}")
        


world_size = 8

if __name__ == "__main__":
    mp.spawn(wrapped_run_function, nprocs=world_size, join=True)

