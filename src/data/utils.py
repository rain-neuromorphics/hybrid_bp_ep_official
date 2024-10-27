import torchvision
import torch
from typing import Optional
from src.data.imagenet32 import ImageNet32Dataset
import os
from torch.utils.data import DataLoader, DistributedSampler, BatchSampler
from hydra.utils import get_original_cwd
from typing import Tuple

def getDataLoaders(
    dataset: str,
    is_mse: bool = False,
    batch_size: int = 256,
    directory: Optional[str] = None,
    output_size: int = 1000,
    world_size: int = 1,
    rank: int = -1,
    **kwargs
) -> Tuple[DataLoader, DataLoader]:
    """
    Function which returns the train and validation sets
    
    Args:
        dataset (str): name of the dataset ("mnist", "cifar10", "cifar100" or "imagenet32")
        is_mse (bool): indicates if the MSE loss function is used (in which case targets are one-hot encoded)
        batch_size (int): size of the data batch
        directory (str): directory where the datasets can be found
        output_size (int): number of classes
        world_size (int): number of devices used in parallel (anticipating on future uses of the codebase for data parallelized simulations)
        rank (int): local rank (anticipating on future uses of the codebase for data parallelized simulations)
        
    Returns:
        train_loader (DataLoader), test_loader (DataLoader): train and validation dataloaders. 
    """
    
    class ReshapeTransformTarget:
        def __init__(self, number_classes):
            self.number_classes = number_classes
        
        def __call__(self, target):
            target=torch.tensor(target).unsqueeze(0).unsqueeze(1)
            target_onehot = torch.zeros((1,self.number_classes))      
            return target_onehot.scatter_(1, target, 1).squeeze(0)
        
    if not directory:
        directory = get_original_cwd() + '/src/data/datasets/'
        if not os.path.exists(directory):
            os.mkdir(directory)
    directory += dataset
    
    if dataset=='MNIST':
        if not os.path.exists(directory):
            os.mkdir(directory)
                
        transform = torchvision.transforms.Compose([
            torchvision.transforms.Pad(padding=(2, 2, 2, 2), fill=0),
            torchvision.transforms.RandomCrop(size=[32,32], padding=2, padding_mode='edge'),
            torchvision.transforms.ToTensor()

        ])
        
        mnist_train = torchvision.datasets.MNIST(
            directory,
            train=True,
            transform=transform,
            target_transform=ReshapeTransformTarget(10) if is_mse else None,
            download=True
        )

        mnist_test = torchvision.datasets.MNIST(
            directory,
            train=False,
            transform=transform,
            target_transform=ReshapeTransformTarget(10) if is_mse else None,
            download=True
        )

        train_loader = DataLoader(mnist_train, batch_size = batch_size, shuffle = True, num_workers = 0, drop_last = True)
        test_loader = DataLoader(mnist_test, batch_size = batch_size, shuffle = False, num_workers = 0, drop_last = True)

    elif dataset=='CIFAR10':
        if not os.path.exists(directory):
            os.mkdir(directory)
                
        transform_train = torchvision.transforms.Compose([
            torchvision.transforms.RandomHorizontalFlip(0.5),
            torchvision.transforms.RandomCrop(size=[32, 32], padding=4, padding_mode='edge'),
            torchvision.transforms.ToTensor(), 
            torchvision.transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), 
                                            std=(3*0.2023, 3*0.1994, 3*0.2010)) 
        ])

        transform_test = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                        torchvision.transforms.Normalize(mean=(0.4914, 0.4822, 0.4465),
                                                                                        std=(3*0.2023, 3*0.1994, 3*0.2010)) ])

        cifar10_train = torchvision.datasets.CIFAR10(
            directory,
            train=True,
            transform=transform_train,
            target_transform=ReshapeTransformTarget(10) if is_mse else None,
            download=True
        )

        cifar10_test = torchvision.datasets.CIFAR10(
            directory,
            train=False,
            transform=transform_test,
            target_transform=ReshapeTransformTarget(10) if is_mse else None,
            download=True
        )

        train_loader = DataLoader(cifar10_train, batch_size = batch_size, shuffle = True, num_workers = 0, drop_last = True)
        test_loader = DataLoader(cifar10_test, batch_size = batch_size, shuffle = False, num_workers = 0, drop_last = True)
    
    elif dataset=='CIFAR100':
        if not os.path.exists(directory):
            os.mkdir(directory)
                
        transform_train = torchvision.transforms.Compose([
            torchvision.transforms.RandomHorizontalFlip(0.5),
            torchvision.transforms.RandomCrop(size=[32, 32], padding=4, padding_mode='edge'),
            torchvision.transforms.ToTensor(), 
            torchvision.transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), 
                                            std=(3*0.2023, 3*0.1994, 3*0.2010)) 
        ])

        transform_test = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                        torchvision.transforms.Normalize(mean=(0.4914, 0.4822, 0.4465),
                                                                                        std=(3*0.2023, 3*0.1994, 3*0.2010)) ])

        cifar100_train = torchvision.datasets.CIFAR100(
            directory,
            train=True,
            transform=transform_train,
            target_transform=ReshapeTransformTarget(10) if is_mse else None,
            download=True
        )

        cifar100_test = torchvision.datasets.CIFAR100(
            directory,
            train=False,
            transform=transform_test,
            target_transform=ReshapeTransformTarget(10) if is_mse else None,
            download=True
        )

        train_loader = DataLoader(cifar100_train, batch_size = batch_size, shuffle = True, num_workers = 0, drop_last = True)
        test_loader = DataLoader(cifar100_test, batch_size = batch_size, shuffle = False, num_workers = 0, drop_last = True)
        
    elif dataset=='imagenet32':
        if not os.path.exists(directory):
            os.mkdir(directory)
                
        nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, world_size])  # number of workers
        mean = [0.485, 0.456, 0.406]
        std = [1.5 * 0.229, 1.5 * 0.224, 1.5 * 0.225]
        final_transform = torchvision.transforms.Normalize(mean, std)

        train_transform = torchvision.transforms.Compose([
            torchvision.transforms.RandomHorizontalFlip(0.5),
            torchvision.transforms.RandomCrop(size=[32, 32], padding=4, padding_mode='edge'),
            torchvision.transforms.ToTensor(),
            final_transform
        ])

        test_transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            final_transform
        ])

        # Download training data
        training_data = ImageNet32Dataset(
            # root = "/mnt/disks/disk-imagenet32",
            root = directory,
            train = True,
            transform = train_transform,
            download = True,
        )

        # Download test data
        test_data = ImageNet32Dataset(
            # root = "/mnt/disks/disk-imagenet32",
            root = directory,
            train = False,
            transform = test_transform,
            download = True,
        )
        
        if rank > -1:
            train_sampler = DistributedSampler(training_data)
            train_batch_sampler = BatchSampler(train_sampler, batch_size // nw, drop_last=True)

            train_loader = DataLoader(
                training_data,
                batch_sampler=train_batch_sampler,
                pin_memory=True,
                num_workers=nw,
            )           
        else:
            train_loader = DataLoader(
                training_data,
                batch_size=batch_size,
                shuffle=True,
                pin_memory=True,
                num_workers=0
            )
        # test_data = IndexedDataset(test_data)
        test_loader = DataLoader(
            test_data,
            batch_size = batch_size,
            shuffle = False,
            num_workers = 0,
            sampler = DistributedSampler(test_data, shuffle=False, drop_last=True) if rank > -1 else None
        )
        
    else:
        print('\n Dataset {} currently unsupported'.format(dataset))
        return

    return train_loader, test_loader