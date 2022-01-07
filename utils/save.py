import torch
import os
import pandas as pd
import numpy as np
import json

def save_config(config):
    os.makedirs(config['output']['path'], exist_ok=True)
    path = os.path.join(config['output']['path'], 'config.json')
    with open(path, 'w') as f:
        json.dump(config, f)

def save_out_target(config, i, metrics):
    tm = metrics['train']
    vm = metrics['val']
    t_ep = np.array((tm.predicted, tm.ground_truth))
    v_ep = np.array((vm.predicted, vm.ground_truth))
    os.makedirs(config['output']['path'], exist_ok=True)
    t_save_path = os.path.join(config['output']['path'],'t_{}.npy'.format(i))
    v_save_path = os.path.join(config['output']['path'],'v_{}.npy'.format(i))
    np.save(t_save_path, t_ep)
    np.save(v_save_path, v_ep)

def save_checkpoint(config, model, optimizer, scheduler, epoch, train_batch_losses, val_batch_losses, metrics):
    tm = metrics['train']
    vm = metrics['val']
    def Average(lst):
        return sum(lst) / len(lst)
    # save paths
    save_path = os.path.join(config['output']['path'], config['output']['name'])
    os.makedirs(save_path, exist_ok=True)
    # save model
    model.eval()
    s_dict = {
        'model_state_dict': model.state_dict(), #'model_state_dict': self.model.model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'epoch': epoch,
        'train_loss_mean': Average(train_batch_losses),
        'train_batch_losses': train_batch_losses,
        'train_acc': np.mean(tm.accuracy),
        'val_loss_mean': Average(val_batch_losses),
        'val_batch_losses': val_batch_losses,
        'val_acc': np.mean(vm.accuracy)
        }
    torch.save(s_dict, save_path+'epoch_{epoch}.pth'.format(epoch=epoch))

def save_df(config, df, i, train_batch_losses, val_batch_losses, metrics):
    tm = metrics['train']
    vm = metrics['val']
    def Average(lst):
        return sum(lst) / len(lst)
    s_dict = {
        'epoch': i,
        'train_loss_mean': Average(train_batch_losses),
        'train_batch_losses': train_batch_losses,
        'train_TP': tm.tp.detach().cpu().numpy(), 
        'train_TN': tm.tn.detach().cpu().numpy(), 
        'train_FP': tm.fp.detach().cpu().numpy(),
        'train_FN': tm.fn.detach().cpu().numpy(), 
        'train_acc': tm.acc.detach().cpu().numpy(),
        'train_prec': tm.pre.detach().cpu().numpy(), 
        'train_rec': tm.rec.detach().cpu().numpy(),
        'train_f1': tm.f1.detach().cpu().numpy(),
        'val_loss_mean': Average(val_batch_losses),
        'val_batch_losses': val_batch_losses,
        'val_TP': vm.tp.detach().cpu().numpy(), 
        'val_TN': vm.tn.detach().cpu().numpy(),
        'val_FP': vm.fp.detach().cpu().numpy(), 
        'val_FN': vm.fn.detach().cpu().numpy(),
        'val_acc': vm.acc.detach().cpu().numpy(), 
        'val_prec': vm.pre.detach().cpu().numpy(),
        'val_rec': vm.rec.detach().cpu().numpy(),
        'val_f1': vm.f1.detach().cpu().numpy()
        }
    for key,value in s_dict.items():
	    print(key, ':', value)
    df = df.append(s_dict, ignore_index=True).reset_index(drop=True)
    save_path = os.path.join(config['output']['path'],'df.csv')
    os.makedirs(config['output']['path'], exist_ok=True)
    df.to_csv(save_path, index=False)
    return(df)


def save_all(config, df, model, optimizer, scheduler, epoch, train_batch_losses, val_batch_losses, metrics):
    # save model
    model.eval()
    # metrics
    tm = metrics['train']
    vm = metrics['val']
    def Average(lst):
        return sum(lst) / len(lst)
    # dictionary to save
    s_dict = {
        'epoch': epoch,
        'train_loss_mean': Average(train_batch_losses),
        'train_batch_losses': train_batch_losses,
        'train_TP': tm.tp.detach().cpu().numpy(), 
        'train_TN': tm.tn.detach().cpu().numpy(), 
        'train_FP': tm.fp.detach().cpu().numpy(),
        'train_FN': tm.fn.detach().cpu().numpy(), 
        'train_acc': tm.acc.detach().cpu().numpy(),
        'train_prec': tm.pre.detach().cpu().numpy(), 
        'train_rec': tm.rec.detach().cpu().numpy(),
        'train_f1': tm.f1.detach().cpu().numpy(),
        'val_loss_mean': Average(val_batch_losses),
        'val_batch_losses': val_batch_losses,
        'val_TP': vm.tp.detach().cpu().numpy(), 
        'val_TN': vm.tn.detach().cpu().numpy(),
        'val_FP': vm.fp.detach().cpu().numpy(), 
        'val_FN': vm.fn.detach().cpu().numpy(),
        'val_acc': vm.acc.detach().cpu().numpy(), 
        'val_prec': vm.pre.detach().cpu().numpy(),
        'val_rec': vm.rec.detach().cpu().numpy(),
        'val_f1': vm.f1.detach().cpu().numpy()
    }
    # print values
    for key,value in s_dict.items():
	    print(key, ':', value)
    # ensure save paths are okay
    # model
    model_save_path = os.path.join(config['output']['path'], config['output']['name'])
    os.makedirs(model_save_path, exist_ok=True)
    # df
    os.makedirs(config['output']['path'], exist_ok=True)
    df_save_path = os.path.join(config['output']['path'],'df.csv')
    # numpy
    t_save_path = os.path.join(config['output']['path'],'t_{}.npy'.format(epoch))
    v_save_path = os.path.join(config['output']['path'],'v_{}.npy'.format(epoch))
    t_ep = np.array((tm.predicted, tm.ground_truth))
    v_ep = np.array((vm.predicted, vm.ground_truth))
    np.save(t_save_path, t_ep)
    np.save(v_save_path, v_ep)
    # append & savedf
    df = df.append(s_dict, ignore_index=True).reset_index(drop=True)
    df.to_csv(df_save_path, index=False)
    # add model saves
    s_dict['model_state_dict'] = model.state_dict() #'model_state_dict': self.model.model.state_dict(),
    s_dict['optimizer_state_dict'] = optimizer.state_dict()
    s_dict['scheduler_state_dict'] = scheduler.state_dict()
    # save model
    torch.save(s_dict, model_save_path+'epoch_{epoch}.pth'.format(epoch=epoch))
    return(df)
