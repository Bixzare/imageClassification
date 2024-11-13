import os
import lightning as pl
import torch
import torchvision
from sklearn.metrics import roc_curve, precision_recall_curve
import numpy as np
import torchmetrics
import pandas as pd
# import sys
# sys.path.append("/teamspace/studios/this_studio/mosquito-classification/package/package_model")
#print(os.path.exists(config_path))
from config import Args
from copy import deepcopy
from itertools import product
from collections import OrderedDict
import wandb
from torchvision.models import EfficientNet_B3_Weights
from pytorch_lightning.callbacks import ModelCheckpoint
from torch import nn
import torchvision.models as models


# Class to define the classifiers
class Classifier(torch.nn.Module):

    def __init__(self, args:Args):
        super().__init__()

        # TODO: create models
        """
        self.model = torchvision.models.efficientnet_b3(pretrained=True)
        num_classes = 4 # à ajuster selon ton problème
        # Modifier la dernière couche (classifier) pour correspondre au nombre de classes
        in_features = self.model.classifier[1].in_features  # Récupérer les caractéristiques d'entrée de la couche de classification
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=0.3, inplace=True),
            nn.Linear(in_features, num_classes)  # Créer une nouvelle couche linéaire avec le bon nombre de classes
        )
        """
        """
        # Geler les couches intermédiaires pour l'entraînement
        for param in self.model.parameters():
            param.requires_grad = False  # Geler toutes les couches

        # Ne dégeler que la dernière couche (classifier)
        for param in self.model.classifier.parameters():
            param.requires_grad = True
        """
        
        self.model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
        self.model.classifier = nn.Linear(self.model.classifier.in_features, 4)
       

    def forward(self,image:torch.Tensor):

        return self.model(image)

# Focal loss function
class FocalLoss(torch.nn.Module):

    def __init__(self,gamma:float,alpha:float):
        super().__init__()
        self.loss_ce = torch.nn.CrossEntropyLoss(reduction='none')
        self.gamma = torch.Tensor([gamma])
        self.alpha = torch.Tensor([alpha])

    def forward(self,logits,targets):
        ce_loss = torch.nn.functional.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (self.alpha * (1-pt)**self.gamma * ce_loss).mean()
        return focal_loss

# -- Pytorch lightning model
class Orchestrator(pl.LightningModule):
    """Class which handles the model training routine using Pytorch Lightning
    """
    def __init__(self, args:Args):
        """Constructor.

        Args:
            args (Args): arguments parsed from Args.py which contains the configuration of the experiment.

        """
        super().__init__()

        
        # config
        self.args = args

        #-- placeholder
        self.test_step_outputs = []

        # self.threshold_prediction = args.threshold_prediction if hasattr(args, 'threshold_prediction') else 0.5
        
        #-- Metrics
        num_classes = self.args.num_classes # number of classes on which to evaluate the models
        mode = 'multiclass' # metrics computed on the leafnode
        """
        self.f1score_train_micro = torchmetrics.F1Score(task=mode,num_classes=num_classes,average='micro',
                                                        threshold=self.threshold_prediction)
        self.f1score_valid_micro = torchmetrics.F1Score(task=mode,num_classes=num_classes,average='micro',
                                                        threshold=self.threshold_prediction)
        self.f1score_test_micro = torchmetrics.F1Score(task=mode,num_classes=num_classes,average='micro',
                                                     threshold=self.threshold_prediction)
        """
        self.f1score_train_macro = torchmetrics.F1Score(task=mode,num_classes=num_classes,average='macro')
        self.f1score_valid_macro = torchmetrics.F1Score(task=mode,num_classes=num_classes,average='macro')
        self.f1score_test_macro = torchmetrics.F1Score(task=mode,num_classes=num_classes,average='macro')
        """
        self.accuracy_train_micro = torchmetrics.Accuracy(task=mode,num_classes=num_classes,average='micro',
                                                          threshold=self.threshold_prediction)
        self.accuracy_valid_micro = torchmetrics.Accuracy(task=mode,num_classes=num_classes,average='micro',
                                                          threshold=self.threshold_prediction)
        self.accuracy_test_micro = torchmetrics.Accuracy(task=mode,num_classes=num_classes,average='micro',
                                                         threshold=self.threshold_prediction)
        """
        self.accuracy_train_macro = torchmetrics.Accuracy(task=mode,num_classes=num_classes,average='macro',
                                                          )
        self.accuracy_valid_macro = torchmetrics.Accuracy(task=mode,num_classes=num_classes,average='macro',
                                                          )
        self.accuracy_test_macro = torchmetrics.Accuracy(task=mode,num_classes=num_classes,average='macro',
                                                         )

        self.aucpr_train_macro = torchmetrics.AveragePrecision(task=mode,num_classes=num_classes,average='macro',thresholds=5)
        self.aucpr_valid_macro = torchmetrics.AveragePrecision(task=mode,num_classes=num_classes,average='macro',thresholds=5)
        
        self.confmatrix_train = torchmetrics.StatScores(task=mode,num_classes=num_classes,average=None,
                                                        multidim_average='global')
        self.confmatrix_valid = torchmetrics.StatScores(task=mode,num_classes=num_classes,average=None,
                                                        multidim_average='global')
        self.confmatrix_test = torchmetrics.StatScores(task=mode,num_classes=num_classes,average=None,
                                                       multidim_average='global')

        #- these two are not used
        # self.aucpr_train_micro = torchmetrics.AveragePrecision(task=mode,average='micro',thresholds=5)
        # self.aucpr_valid_micro = torchmetrics.AveragePrecision(task=mode,average='micro',thresholds=5)

        self.metrics_objects = dict()
        self.metrics_objects['train'] = [['f1_score_macro',self.f1score_train_macro],
                                         #['f1_score_micro',self.f1score_train_micro],
                                        ['accuracy_macro',self.accuracy_train_macro],
                                       #  ['accuracy_micro',self.accuracy_train_micro],
                                         ['averagePrecision_macro',self.aucpr_train_macro],
                                        #  ['averagePrecision_micro',self.aucpr_train_micro],
                                         ]   
        self.metrics_objects['valid'] = [['f1_score_macro',self.f1score_valid_macro],
                                        # ['f1_score_micro',self.f1score_valid_micro], 
                                         ['accuracy_macro',self.accuracy_valid_macro],
                                         #['accuracy_micro',self.accuracy_valid_micro],
                                         ['averagePrecision_macro',self.aucpr_valid_macro],
                                        #  ['averagePrecision_micro',self.aucpr_valid_micro],
                                         ]
        self.metrics_objects['test'] = [['f1_score_macro',self.f1score_test_macro],
                                         #['f1_score_micro',self.f1score_test_micro], 
                                         ['accuracy_macro',self.accuracy_test_macro],
                                         #['accuracy_micro',self.accuracy_test_micro],
                                         ]
        
        #-- Get model
        self.model = Classifier(args=Args)
        
        # -- declare loss functions
        self.loss_ce = torch.nn.CrossEntropyLoss(reduction='mean')
        self.loss_focal = FocalLoss(gamma=self.args.focal_loss_gamma,alpha=self.args.focal_loss_alpha)
   
    def forward(self, image:torch.Tensor)->torch.Tensor:
        """Runs forward function of the neural network

        Args:
            image (torch.Tensor): image

        Returns:
            tuple: logits from neural network
        """
        return self.model(image)

    
    def check_inputs(self,images:torch.Tensor,labels:torch.Tensor):
        # Vérification de la forme des images
        if images.ndim != 4:
            raise ValueError(f"Expected images to have 4 dimensions (batch_size, channels, height, width), but got {images.ndim} dimensions.")
        
        # Vérification des types de données des images
        if not torch.is_tensor(images):
            raise TypeError("Expected images to be of type torch.Tensor.")
        
        # Vérification de la forme des cibles
        if labels.ndim != 1:
            raise ValueError(f"Expected labels to have 1 dimension (batch_size), but got {labels.ndim} dimensions.")

        # Vérification des types de données des cibles
        if not torch.is_tensor(labels):
            raise TypeError("Expected labels to be of type torch.Tensor.")
        
        # Vérification des types des étiquettes
        if labels.dtype != torch.long:
            raise ValueError(f"Expected labels to be of type torch.long, but got {labels.dtype}.")

        # Vérification que les étiquettes sont dans la plage correcte
        if not (labels.min() >= 0 and labels.max() < self.args.num_classes):
            raise ValueError(f"Labels should be between 0 and {self.args.num_classes - 1}, but got min: {labels.min().item()} and max: {labels.max().item()}.")

        # Optionnel : Vérification de la taille des images et des cibles
        #print(f"Images shape: {images.shape}, Labels shape: {labels.shape}")

    def get_loss(self,logits_labels:torch.Tensor,labels:torch.Tensor):
        """Computes loss

        Args:
            logits_labels (torch.Tensor): predicted logits
            labels (torch.Tensor): targets  

        Returns:
            torch.Tensor: loss - else
        """
        loss = 0
        if self.args.criterion == "focal":
            loss = self.loss_focal(logits_labels, labels.long())

        elif self.args.criterion == "ce":
            loss = self.loss_ce(logits_labels, labels.long())
            
        else:
            raise NotImplementedError
        
        return loss        

    def shared_step(self, batch:tuple, stage:str, batch_idx:int):
        """Runs computes loss, predicted labels, and evaluation metrics for a given batch

        Args:
            batch (tuple): (images,labels)
            stage (str): either 'train', 'valid' or 'test'
            batch_idx (int): batch index in dataloader

        Returns:
            torch:Tensor: loss
        """
        images, labels = batch
        labels = labels.long()
        
        # -- checks
        self.check_inputs(images,labels)

        # -- forward pass
        logits_labels = self.forward(images)
        
        # -- get class
        prob_labels = torch.softmax(logits_labels, dim=1) # TODO: review
        # print(prob_labels.shape)
        # Log the unique predicted labels
        #print(f"Unique predicted labels: {torch.unique(prob_labels)}")
        
        # -- compute loss
        loss = self.get_loss(logits_labels,labels)

        # -- log metrics
        for name,metric in self.metrics_objects[stage]:                    
            metric.update(prob_labels, labels)
            self.log(f'{name}_{stage}',
                        metric,
                        on_epoch=True,
                        on_step=False,
                        logger=True,
                        prog_bar=False)

        # -- Log loss
        self.log(f"{stage}_loss",
            loss.item(),
            prog_bar=False,
            on_step=False,
            on_epoch=True,
            logger=True)

        if stage == 'test':
            self.confmatrix_test.update(prob_labels,labels)
                
        return loss

    def shared_epoch_end(self, outputs:list[tuple], stage:str):
        """Computes metrics at the end of an epoch

        Args:
            outputs (list): a list of tuple containing pairs of (predicted probability of label, predicted label)
            stage (str): either 'train', 'valid' or 'test'

        """
        if len(outputs) == 0:
            print(f"No outputs found for stage: {stage}")
            return 
        
        if stage == 'test':
            evaluation_labels = self.args.label_indices
            evaluation_label_names = self.args.label_names

            # Log confusion matrix on labels
            probs = np.vstack([prob for prob,label in outputs])
            labels = np.hstack([label for prob,label in outputs])
            wandb.log({f"conf_mat_{stage}":wandb.plot.confusion_matrix(y_true=labels,
                                                                        probs=probs,
                                                                       class_names=evaluation_label_names)})

            # log metris for each leaf label
            confmatrix = self.confmatrix_test.compute().cpu().numpy()
            self.confmatrix_test.reset()
            accuracy = lambda tp,fp,tn,fn : round((tp+tn)/(tp+fp+tn+fn),4)
            f1_score = lambda tp,fp,tn,fn : round(2*tp/(2*tp+fp+fn),4)
            try:
                data = dict()
                cols = ['label','accuracy','f1_score','num_examples','label_name']
                for ind,label in enumerate(evaluation_labels):
                    acc = accuracy(*confmatrix[ind,:-1].tolist())
                    f1 = f1_score(*confmatrix[ind,:-1].tolist())
                    num_examples = confmatrix[ind,-1]
                    label_name = self.labels_names.at[label,'name']
                    data[ind] = [label,acc,f1,num_examples,label_name]
                data = pd.DataFrame.from_dict(data, orient='index',columns=cols)
                wandb.log({f'PerClassMetrics_{stage}_epoch_{self.current_epoch}':wandb.Table(dataframe=data)})
                                
            except Exception as e:
                print('The metrics could not be computed -> ',e)

    def training_step(self, batch:tuple, batch_idx:int):   
        """Runs a training step

        Args:
            batch (tuple): (images,labels)
            batch_idx (int): batch index in dataloader

        Returns:
            torch.Tensor: loss
        """
        return self.shared_step(batch, "train", batch_idx)

    def validation_step(self, batch:tuple, batch_idx:int):
        """Runs a validation step

        Args:
            batch (tuple): (images,labels)
            batch_idx (int): batch index in dataloader

        Returns:
            torch.Tensor: loss
        """
        return self.shared_step(batch, "valid", batch_idx)

    def test_step(self, batch:tuple, batch_idx:int):
        """Runs a testing step 

        Args:
            batch (tuple): (images,labels)
            batch_idx (int): batch index in dataloader

        Returns:
            torch.Tensor: loss
        """
        return self.shared_step(batch, "test",batch_idx)
    
    def on_test_epoch_end(self):
        """Computes specific evaluation metrics at the end of an epoch
        """
        self.shared_epoch_end(self.test_step_outputs, "test")
        self.test_step_outputs.clear()

    def predict_step(self, batch, batch_idx:int):
        """Prediction step on a batch

        Args:
            batch (tuple or torch.Tensor): (images,...) or images
            batch_idx (int): batch index in dataloader

        Returns:
            tuple: (predicted logits, embeddings)
        """
        if isinstance(batch,tuple) or isinstance(batch,list):
            images,_ = batch
        else:
            images = batch
        
        logits_labels = self.forward(images)
        
        return logits_labels

    def configure_optimizers(self):
        """Configured the optimizers and learning schedulers

        Returns:
            dict: information of optimizer used and learning schedulers
        """
        opt = torch.optim.Adam(
            self.parameters(),
            lr=self.args.lr,
            weight_decay=self.args.weight_decay)

        if self.args.optimizer == "SGD":
            opt = torch.optim.SGD(
                self.parameters(),
                lr =self.args.lr,
                weight_decay=self.args.weight_decay,
                momentum=0.9,
                nesterov=True)

        # --- Default
        interval = 'epoch'
        sch = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt,
            mode=self.args.lr_scheduler_mode,
            factor=self.args.lr_scheduler_factor,
            patience=self.args.lr_scheduler_patience,
            min_lr=self.args.min_lr)

        if self.args.lr_scheduler == "CosineAnnealingWarmRestarts":
            sch = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                opt, T_0=self.args.T_0, T_mult=self.args.T_mult, eta_min=self.args.min_lr)
            
        elif self.args.lr_scheduler == "CosineAnnealingLR":
            sch = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt, T_max=self.args.max_epochs, eta_min=self.args.min_lr)

        elif self.args.lr_scheduler == "MultiplicativeLR":
            lr_func = lambda epoch: self.args.exp_lr_gamma*(epoch<=self.args.T_0) + 1.0*(epoch>self.args.T_0)
            sch = torch.optim.lr_scheduler.MultiplicativeLR(
                opt,
                lr_lambda= lr_func,
                last_epoch=-1)

        elif self.args.lr_scheduler == "OneCycleLR":
            sch = torch.optim.lr_scheduler.OneCycleLR(
                opt,
                max_lr=1e-3,
                total_steps=self.trainer.estimated_stepping_batches,
                anneal_strategy='cos')
            interval = 'step'

        else:
            print('using default scheduler: ReduceOnPlateau')
        out = {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": sch,
                "monitor": self.args.metric_to_monitor_lr_scheduler,
                "frequency": 1,
                "interval": interval
            },
        }

        return out




