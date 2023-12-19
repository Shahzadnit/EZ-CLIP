
import torch

def epoch_saving(epoch, model, optimizer, filename):
    torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    }, filename) #just change to your preferred folder/filename

def best_saving(working_dir, epoch, model, optimizer, data_for):
    best_name = '{}/model_best_{}.pt'.format(working_dir, data_for)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, best_name)  # just change to your preferred folder/filename