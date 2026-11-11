import torch.optim as optim
import torch
import torch.nn as nn
import mlflow
import numpy as np
from tqdm import tqdm
from src.utils.plotting_utils import plot_est_comp, plot_alpha_comp
import timeit


EMPTY_LOSS_DICT = {
    'MSE': []
}


def l1_loss(pred, target, l1_weight):
    """
    Compute L1 loss;
    l1_weigh default: 0.1
    """
    err = torch.mean(torch.abs(pred - target))
    err = l1_weight * err
    return err


class CNNTrainer():
    def __init__(self, model, Psi, bpdn_est, dictionary, args):
        self.model = model
        self.bpdn_est = bpdn_est
        self.dictionary = dictionary

        self.device = args['device']

        self.load_model_run = args['load_model_run']
        self.load_model_epoch = args['load_model_epoch']
        
        self.batch_size = args['batch_size']
        self.lr = args['lr']        
        self.lr_dec_after = args['lr_dec_after']
        self.lr_dec_every = args['lr_dec_every']

        if self.load_model_epoch != None:
            self.model = mlflow.pytorch.load_model(f'runs:/{self.load_model_run}/models/CNN_ep{self.load_model_epoch}')

        self.model = self.model.to(self.device)

        # set different lr for regularization weights and network weights
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr, weight_decay=0.001)

        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=0.9)  # step-wise

        self.train_loss = nn.MSELoss()


    def preprocess_batch(self, x_in, y_target, x_0, x_bpdn):
        # measured vector (104*1); add channels
        # CIKK: vector b (16) --\/
        x_in = torch.unsqueeze(x_in, 1).float().to(self.device)

        # target image (64*64)
        y_target = torch.unsqueeze(y_target, 1).float().to(self.device)

        return x_in, y_target
    
    
    def train(self, train_loader, valid_loader, epochs, start_epoch=0, log_model_every=10, log_comp_fig_every=10):
        # if start_epoch:
        #     self.load_model(self.start_epoch)

        for epoch in tqdm(range(1 + start_epoch, epochs + start_epoch + 1)):
            losses_dict = {k: [] for k in EMPTY_LOSS_DICT.keys()}

            self.model.train(True)
            for batch_idx, (x_in, y_target, x_0, x_bpdn) in enumerate(train_loader):
                x_in, y_target = self.preprocess_batch(x_in, y_target, x_0, x_bpdn)

                self.model.zero_grad(set_to_none=True)
                self.optimizer.zero_grad()

                pred = self.model(x_in)   # forward

                loss = self.train_loss(pred, y_target)
                losses_dict['MSE'].append(loss.item())

                # backpropagate the gradients
                loss.backward()
                self.optimizer.step()
                
            # log average epoch loss values to MLflow
            for k, v in losses_dict.items():
                mlflow.log_metric('loss_train_' + k, np.mean(v), step=epoch)

            losses_dict = {k: [] for k in EMPTY_LOSS_DICT.keys()}
            self.model.eval()
            with torch.no_grad():
                for batch_idy, (x_in, y_target, x_0, x_bpdn) in enumerate(valid_loader):
                    x_in, y_target = self.preprocess_batch(x_in, y_target, x_0, x_bpdn)
    
                    pred = self.model(x_in)   # forward
    
                    loss = self.train_loss(pred, y_target)
                    losses_dict['MSE'].append(loss.item())
                    
                    # plot validation batch
                    if not epoch % log_comp_fig_every and not batch_idy:
                        plot_est_comp(x_in, torch.zeros(pred.shape), pred, y_target, self.bpdn_est, self.dictionary,
                                      'valid', epoch, batch_idy)
            
                # log average epoch loss values and model parameters to MLflow
                for k, v in losses_dict.items():
                    mlflow.log_metric('loss_valid_' + k, np.mean(v), step=epoch)
        
            if not (epoch % log_model_every) and epoch > 0:
                # with torch.no_grad():
                #     model_signature = mlflow.models.infer_signature(X_train.numpy(), self.model(X_train).numpy())
                mlflow.pytorch.log_model(self.model, artifact_path=f'models/CNN_ep{epoch}')#, signature=model_signature)


    def evaluate(self, test_loader, criterion=None, crit_text=None, mlflow_log=True):
        assert not ((type(criterion) != type(None)) ^ (crit_text != None))
        losses_dict = {k: [] for k in EMPTY_LOSS_DICT.keys()}
        self.model.eval()
        with torch.no_grad():
            for batch_idy, (x_in, y_target, x_0, x_bpdn) in enumerate(test_loader):
                x_in, y_target = self.preprocess_batch(x_in, y_target, x_0, x_bpdn)

                start = timeit.default_timer()
                pred = self.model(x_in)   # forward
                stop = timeit.default_timer()
                print('Time: ', stop - start) 

                if type(criterion) == type(None):
                    loss = self.train_loss(pred, y_target)
                    losses_dict['MSE'].append(loss.item())
                    for k, v in losses_dict.items():
                        if mlflow_log:
                            mlflow.log_metric('loss_test_' + k, np.mean(v))
                        else:
                            print('loss_test_' + k + ': ', np.mean(v))
                else:
                    loss = criterion(pred, y_target)
                    losses_dict['MSE'].append(loss.item())
                    if mlflow_log:
                        mlflow.log_metric(f'loss_test_{crit_text}', loss.item())
                    else:
                        print(f'loss_test_{crit_text}: ', loss.item())


    def infer(self, test_loader):
        res = []
        self.model.eval()
        with torch.no_grad():
            for batch_idy, (x_in, y_target, x_0, x_bpdn) in enumerate(test_loader):
                x_in, y_target = self.preprocess_batch(x_in, y_target, x_0, x_bpdn)
    
                pred = self.model(x_in)   # forward

                res.append(pred)
        return res