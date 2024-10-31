from src.algorithms import getAlgorithm
from src.models import getModel
from src.data.utils import getDataLoaders
from .utils import getCriterion, getOptimizer, getScheduler, train, validate,cleanup, load, save
import torch
import time
from omegaconf import DictConfig, OmegaConf
from typing import Optional
from . import Experiment
from torch.utils.tensorboard import SummaryWriter
import wandb

class TrainingExperiment(Experiment):
    """
    Class implementing a training experiment
    
    Extra attributes (compared to parent class)
    ------------------------------------------
    logger (Optional[str]):
        logger type (if any)
    
    Methods
    -------
    __init__:
        implements parent __init__ method and picks a logger if one if specified
    _run:
        runs the training experiment
    log_hparams_tb:
        logs hyperparameters when tensorboard is used
    """
    
    def __init__(
        self, 
        cfg: DictConfig,
        logger: Optional[str] = None,
        project_name: str = "deep-eqprop",
        **kwargs
        ) -> None:
        """
        Instantiates a TrainingExperiment object
        
        Args:
            cfg (DictConfig):
                hydra configuration which specifies all the needed information to run the training experiment
            logger (str):
                specifies a logger if any is used
            project_name (str):
                name of the project that is used when wandb logger is employed
        """
        
        super().__init__(cfg, **kwargs)
        
        self.logger = None
        if logger:
            if logger == 'wandb':
                try:
                    import wandb
                except ImportError:
                    ImportError("Please install wandb or remove logger = wandb from hydra config file")
                else:
                    logger = wandb
                    logger.init(project=project_name , config=OmegaConf.to_container(self.cfg,resolve=True))
                    self.logger = logger
            elif logger == 'tensorboard':
                try:
                    from torch.utils.tensorboard import SummaryWriter
                except ImportError:
                    ImportError("Please install tensorboard or remove logger = tensorboard from hydra config file")
                else:
                    hparams = OmegaConf.to_container(self.cfg,resolve=True)
                    logger = SummaryWriter("tensorboard/"+project_name+'_'+hparams['model']['name']+'_'+hparams['data']['name'])
                    self.logger = logger
                    self.logger.add_text('Hyperparameters/config',str(hparams))
                    
            else:
                raise ValueError("{} logger currently unsupported".format(logger))

    def _run(self, rank: int = -1, world_size: int = -1) -> None:
        """
        Executes the whole training experiment
        
        Args:
            rank (int):
                local rank (anticipating on future uses of the codebase for data parallelized simulations)
            world_size (int):
                number of devices at use (same remark as above)
        """
        
        if self.device != 'cpu':
            device = torch.device('cuda:'+ str(self.device) if torch.cuda.is_available() else 'cpu')  
        else:
            device = torch.device('cpu')
        
        train_loader, val_loader = getDataLoaders(
            self.cfg.data.name,
            **self.cfg.data.config,
            is_mse = self.cfg.trainer.criterion == "mse",
            world_size = world_size,
            rank = rank
        )
        
        input_size, output_size = self.cfg.data.config.input_size, self.cfg.data.config.output_size
        
        model = getModel(
            self.cfg.model.name,
            self.cfg.model,
            device,
            input_size = input_size,
            num_classes = output_size,
        ).to(device)
                
        algorithm = getAlgorithm(self.cfg.algorithm.name, self.cfg.algorithm.config)
        criterion  = getCriterion(self.cfg.trainer.criterion).to(device)
        optimizer = getOptimizer(model, **self.cfg.trainer.optimizer, rank=rank)
        scheduler = getScheduler(optimizer, self.cfg.trainer.scheduler) if 'scheduler' in self.cfg.trainer is not None else None

        if self.load_dir:
            model, optimizer, scheduler, start_epoch, best = load(model, optimizer, scheduler, self.load_dir)
        else:
            start_epoch, best = 0, 0.

        start_epoch, best = 0, 0.
        top1_t, top5_t, top1_v, top5_v = [], [], [], []
        wct = []

        start_time = time.time()
        
        for epoch in range(start_epoch, self.cfg.trainer.epochs):
            top1_t_, top5_t_, train_loss = train(train_loader, model, algorithm, criterion, device, optimizer, scheduler, rank, epoch)
            top1_v_, top5_v_, val_loss = validate(val_loader, model, criterion, device, rank)

            top1_t += [top1_t_]
            top5_t += [top5_t_]
            top1_v += [top1_v_]
            top5_v += [top5_v_]
            
            wct = time.time() - start_time

            if self.logger:
                if isinstance(self.logger, SummaryWriter):
                    self.logger.add_scalar('train_acc', top1_t_, epoch)
                    self.logger.add_scalar('train5_acc', top5_t_, epoch)
                    self.logger.add_scalar('val_acc', top1_v_, epoch)
                    self.logger.add_scalar('val5_acc', top5_v_, epoch)
                    self.logger.add_scalar('wct', wct, epoch)        
                    self.logger.add_scalar('train_loss', train_loss, epoch)
                    self.logger.add_scalar('val_loss', val_loss, epoch)
                else:
                    self.logger.log({'top1_train': top1_t_, 'top5_train': top5_t_, 'top1_val': top1_v_, 'top5_val': top5_v_, 'wct': wct})

                if self.save and top1_v_ > best:
                    save(model, optimizer, scheduler, epoch + 1, top1_t, top5_t, top1_v, top5_v, wct) 
        
        if rank > -1: cleanup()

    def log_hparams_tb(self, hparams: DictConfig) -> None:
        """
        Logs hyperparameters into Tensorboard
        
        Args:
            hparams (DictConfig): hydra config object containing hyperparameters
        """

        self.logger.add_scalar('Hyperparameters/beta',hparams['algorithm']['config']['beta'])
        self.logger.add_text('Hyperparameters/algorithm',hparams['algorithm']['name'])
        self.logger.add_scalar('Hyperparameters/T2',hparams['algorithm']['config']['T2'])
        self.logger.add_scalar('Hyperparameters/lr',hparams['trainer']['optimizer']['lr'])
        self.logger.add_scalar('Hyperparameters/weight_decay',hparams['trainer']['optimizer']['config']['weight_decay'])
        self.logger.add_scalar('Hyperparameters/tmax',hparams['trainer']['scheduler']['config']['T_max'])
        self.logger.add_scalar('Hyperparameters/eta_min',hparams['trainer']['scheduler']['config']['eta_min'])
        self.logger.add_text('Hyperparameters/data',hparams['data']['name'])
        self.logger.add_scalar('Hyperparameters/batch_size',hparams['data']['config']['batch_size'])
        if hparams['model']['config'].get('norm_pool'):
            self.logger.add_scalar('Hyperparameters/norm_pool',hparams['model']['config']['norm_pool'])
        self.logger.add_text('Hyperparameters/activation',hparams['model']['config']['activation'])
        self.logger.add_scalar('Hyperparameters/T1',hparams['model']['config']['T1'])
        self.logger.add_text('Hyperparameters/init',hparams['model']['config']['normalization']['weights']['name'])
        self.logger.add_scalar('Hyperparameters/V',hparams['model']['config']['normalization']['weights']['config']['V'])
        self.logger.add_text('Hyperparameters/model',hparams['model']['name'])
