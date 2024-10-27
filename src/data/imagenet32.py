from pathlib import Path
import pickle
import shutil
from typing import Callable, ClassVar, Union
import gdown
import numpy as np
from torchvision.datasets.vision import VisionDataset
from PIL import Image
from torchvision.datasets import ImageNet
from typing import Any
import wget
import tarfile

class ImageNet32Dataset(VisionDataset):
    """
    Downsampled ImageNet 32x32 dataset.
    """

    url: ClassVar[str] = "https://drive.google.com/uc?id=1XAlD_wshHhGNzaqy8ML-Jk0ZhAm8J5J_"
    md5: ClassVar[str] = "64cae578416aebe1576729ee93e41c25"
    archive_filename: ClassVar[str] = "imagenet32.tar.gz"

    def __init__(
        self,
        root: Union[str, Path],
        readonly_datasets_dir: Union[str, Path, None] = None,
        train: bool = True,
        transform: Union[Callable, None] = None,
        target_transform: Union[Callable, None] = None,
        download: bool = False,
    ):
        """
        Initializes a new ImageNet32 dataset.
        
        Args:
            root: Path to the root directory.
            readonly_datasets_dir: Path to the readonly datasets directory. 
            train: If True, creates dataset from training set, otherwise dataset from test set.
            transform: Optional transform to be applied on a sample.
            target_transform: Optional transform to be applied on a target.
            download: If True, downloads the dataset from the internet and puts it in the root directory.
        """

        super().__init__(str(root), transform=transform, target_transform=target_transform)
        self.base_folder = "imagenet32"
        self.train = train  # training set or test set
        self.split = "train" if self.train else "val"
        self.split_folder = f"out_data_{self.split}"
        # TODO: Look for the archive in this directory before downloading it.
        self.readonly_datasets_dir = (
            Path(readonly_datasets_dir).expanduser().absolute() if readonly_datasets_dir else None
        )

        self._data_loaded = False
        self.data: np.ndarray
        self.targets: np.ndarray

        if download:
            self._download_dataset()
            self._load_dataset()
        else:
            try:
                self._load_dataset()
            except FileNotFoundError as err:
                raise RuntimeError(
                    f"Missing the files for ImageNet32 {self.split} dataset, run this with "
                    f"`download=True` first."
                ) from err

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        """
        Returns:
            int: The number of items in the dataset.
        """
        return len(self.data)

    def _download_dataset(self) -> None:
        """
        Downloads and extracts the dataset if it doesn't exist.
        """
        archive_path = (Path(self.root) / self.archive_filename).absolute()
        extracted_path = (Path(self.root) / self.base_folder).absolute()
        root_path = Path(self.root).absolute()

        def extract_archive_in_root():
            # Check if the archive is already extracted somehow?
            print(f"Extracting archive {archive_path} to {root_path}")
            shutil.unpack_archive(archive_path, extract_dir=str(root_path))

        if extracted_path.exists():
            print(f"Extraction path {extracted_path} already exists.")
            try:
                self._load_dataset()
                print(f"Archive already downloaded and extracted to {extracted_path}")
            except Exception as exc:
                # Unable to load the dataset, for some reason. Re-extract it.
                print(f"Unable to load the dataset from {extracted_path}: {exc}\n")
                print("Re-extracting the archive, which will overwrite the files present.")
                extract_archive_in_root()
            return

        if archive_path.exists():
            extract_archive_in_root()
            return
        if (
            self.readonly_datasets_dir
            and (self.readonly_datasets_dir / self.archive_filename).exists()
        ):
            readonly_archive_path = self.readonly_datasets_dir / self.archive_filename
            print(f"Found the archive at {readonly_archive_path}")
            print(f"Copying archive from {readonly_archive_path} -> {archive_path}")
            shutil.copyfile(src=readonly_archive_path, dst=archive_path, follow_symlinks=False)
            extract_archive_in_root()
            return

        if not archive_path.exists():
            print(f"Downloading the archive to {archive_path}")
            # TODO: This uses the ~/.cache/gdown/ directory, which is not great!
            gdown.cached_download(
                url=self.url,
                md5=self.md5,
                path=str(archive_path),
                quiet=False,
                postprocess=gdown.extractall,
            )

    def _load_dataset(self):
        """
        Loads the dataset into memory.
        """
        if self._data_loaded:
            print("Data already loaded. Skipping.")
            return

        data = []
        targets = []
        # Load the picked numpy arrays
        print(f"Loading ImageNet32 {self.split} dataset...")
        for i in range(10):
            file_name = "train_data_batch_" + str(i+1)
            file_path = Path(self.root, self.base_folder, self.split_folder, file_name).absolute()
            with open(file_path, "rb") as f:
                entry = pickle.load(f, encoding="latin1")
                data.append(entry["data"])
                if "labels" in entry:
                    targets.extend(entry["labels"])
                else:
                    targets.extend(entry["fine_labels"])
        self.targets = np.array(targets) - 1
        # self.targets = [t - 1 for t in self.targets]
        self.data = np.vstack(data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))
        print(f"Loaded {len(self.data)} images from ImageNet32 {self.split} split")
        self._data_loaded = True
