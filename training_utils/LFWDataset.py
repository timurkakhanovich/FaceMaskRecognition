from pathlib import Path
from typing import Optional, TypeVar, Tuple, Any, List, Dict

import torch
from torch.utils.data import Dataset
import torchvision
import os
import numpy as np
import multiprocessing
import glob
from PIL import Image

VisionTransforms = TypeVar("VisionTransforms")


# Image Loader  
class TripletTrainingDataset(Dataset):
    def __init__(
        self,
        root_dir: Path,
        num_triplets: int,
        classes: int = 100,
        batch_size: int = 1, 
        num_identities_per_batch: int = 32,
        transform: Optional[VisionTransforms] = None,
        num_workers: int = 0,
    ):
        self.data_path = root_dir
        self.data_dir = list(map(
            lambda cur_folder: os.path.join(self.data_path, cur_folder), 
            sorted(os.listdir(self.data_path))
        ))

        self.num_workers = num_workers if num_workers != 0 else os.cpu_count()

        self.num_triplets = num_triplets
        self.batch_size = batch_size
        self.num_identities_per_batch = num_identities_per_batch
        self.transform = transform
        self.classes = classes
        self.triplets = self.multiprocess_generate_triplets()

    def generate_triplets(self, num_triplets_per_process: int, process_id: int) -> None:
        self.set_temp_file()

        randomstate = np.random.RandomState(seed=42)
        labels = np.load('Data/labels.npy')

        num_iterations_per_epoch = num_triplets_per_process // self.batch_size

        triplets = np.zeros((num_iterations_per_epoch, self.batch_size, 5))
        for iter_idx in range(num_iterations_per_epoch):   
            triplets_within_batch = []
            classes_subset_for_triplets = randomstate.choice(self.classes, self.num_identities_per_batch)  # per batch  

            for _ in range(self.batch_size):
                pos_class = randomstate.choice(classes_subset_for_triplets)

                while True:
                    neg_class = randomstate.choice(classes_subset_for_triplets)

                    if pos_class != neg_class:
                        break

                ianc, ipos = randomstate.choice(labels[pos_class], 2, replace=False)
                ineg = randomstate.choice(labels[neg_class])

                triplets_within_batch.append([
                    pos_class,
                    neg_class,
                    ianc,
                    ipos,
                    ineg
                ])

            triplets[iter_idx] = np.array(triplets_within_batch)

        save_path = Path(
            f"Datasets/temp/temp_training_triplets_identities_{self.num_identities_per_batch}"
            f"_batch_{self.batch_size}_process_{process_id}.npy"
        )
        with open(save_path) as fin:
            np.save(fin, triplets)
    
    def multiprocess_generate_triplets(self) -> None:
        num_triplets_per_process = self.num_triplets // self.num_workers

        processes = []
        for process_id in range(self.num_workers):
            processes.append(multiprocessing.Process(
                target=self.generate_triplets,
                args=(num_triplets_per_process, process_id + 1)
            ))
        
        for process in processes:
            process.start()
        
        for process in processes:
            process.join()
        
        process_files = glob.glob("temp/*.npy")

        total_triplets = []
        for current_file in process_files:
            total_triplets.append(np.load(current_file).astype(int))
        
        return np.vstack(total_triplets)
    
    def get_triplet_by_indices(
        self,
        pos_class: int,
        neg_class: int,
        ianc: int,
        ipos: int,
        ineg: int,
    ) -> Dict[str, Any]:
        pos_dir = list(self.data_dir[pos_class].glob("*"))
        neg_dir = list(self.data_dir[neg_class].glob("*"))

        anc_data = pos_dir[ianc]
        pos_data = pos_dir[ipos]
        neg_data = neg_dir[ineg]

        anc = Image.open(self.data_dir[pos_class] / anc_data)
        pos = Image.open(self.data_dir[pos_class] / pos_data)
        neg = Image.open(self.data_dir[neg_class] / neg_data)

        if self.transform:
            anc = self.transform(anc)
            pos = self.transform(pos)
            neg = self.transform(neg)
        
        return { 
            "anc_img": anc, 
            "pos_img": pos, 
            "neg_img": neg,
            "pos_class": pos_class,
            "neg_class": neg_class 
        }

    def __getitem__(self, index: int) -> List[Any]:
        batch = self.triplets[index]

        batch_sample = []
        for data_info in batch:
            batch_sample.append(self.get_triplet_by_indices(*data_info))

        return batch_sample
    
    def __len__(self) -> int:
        return len(self.triplets)


class TriletValidatingDataset(torchvision.datasets.ImageFolder):
    def __init__(
        self,
        root_dir: Path,
        pairs_path: Path,
        transform: Optional[VisionTransforms] = None,
    ) -> None:

        super().__init__(root_dir, transform)

        self.pairs_path = pairs_path

        # LFW dir contains 2 folders: faces and lists
        self.validation_images = self.get_lfw_paths(root_dir)

    def read_lfw_pairs(self, pairs_filename: Path) -> np.ndarray:
        pairs = []
        
        with open(pairs_filename, "r") as f:
            for line in f.readlines()[1:]:
                pair = line.strip().split()
                pairs.append(pair)

        return np.array(pairs, dtype=object)

    def get_lfw_paths(self, lfw_dir: Path) -> List[Tuple[Path]]:
        pairs = self.read_lfw_pairs(self.pairs_path)

        nrof_skipped_pairs = 0
        path_list = []
        issame_list = []
        for pair in pairs:
            if len(pair) == 3:
                path0 = lfw_dir / pair[0] / pair[1]
                path1 = lfw_dir / pair[0] / pair[2]
                issame = True
            elif len(pair) == 4:
                path0 = lfw_dir / pair[0] / pair[1]
                path1 = lfw_dir / pair[2] / pair[3]
                issame = False
            if path0.exists() and path1.exists():
                path_list.append((path0, path1, issame))
                issame_list.append(issame)
            else:
                nrof_skipped_pairs += 1
            
        if nrof_skipped_pairs > 0:
            print(f"Skipped {nrof_skipped_pairs} image pairs")
        
        return path_list
    
    def add_extension(self, path: Path) -> Path:
        if (path.with_suffix(".jpg")).exists():
            return path.with_suffix(".jpg")
        elif path.with_suffix(".png"):
            return path.with_suffix(".png")
        else:
            raise RuntimeError(f"No file \"{path}\" with extension png or jpg.")
    
    def __getitem__(self, index: int) -> Tuple[torch.FloatTensor]:
        def transform(img_path):
            img = self.loader(img_path)
            return self.transform(img)
        
        path_1, path_2, issame = self.validation_images[index]
        img1, img2 = transform(path_1), transform(path_2)
        return img1, img2, issame
    
    def __len__(self) -> int:
        return len(self.validation_images)
