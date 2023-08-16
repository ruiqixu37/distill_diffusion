import os
import pytorch_lightning as pl
from PIL import Image
import io
from torch.utils.data.dataloader import default_collate
from torch.utils.data import DataLoader
from functools import partial
import clip
import webdataset as wds
import json


def create_webdataset(
    urls,
    image_transform,
    enable_text=True,
    enable_image=True,
    image_key="jpg",
    caption_key="txt",
    enable_metadata=False,
    cache_path=None,
):
    """Create a WebDataset reader, it can read a webdataset of image, text and json"""
    dataset = wds.WebDataset(urls, cache_dir=cache_path,
                             cache_size=10**10, handler=wds.handlers.warn_and_continue)

    def tokenizer(text):
        return clip.tokenize([text], truncate=True)[0]

    def filter_dataset(item):
        if enable_text and caption_key not in item:
            return False
        if enable_image and image_key not in item:
            return False
        if enable_metadata and "json" not in item:
            return False

        # filter nsfw images
        if enable_metadata:
            metadata_file = json.loads(item["json"])
            if metadata_file["NSFW"] == "NSFW":
                return False

        return True

    filtered_dataset = dataset.select(filter_dataset)

    def preprocess_dataset(item):
        output = {}
        if enable_image:
            image_data = item[image_key]
            image = Image.open(io.BytesIO(image_data))
            image_tensor = image_transform(image)
            output["image_filename"] = item["__key__"]
            output["image_tensor"] = image_tensor

        if enable_text:
            text = item[caption_key]
            caption = text.decode("utf-8")
            tokenized_text = tokenizer(caption)
            output["text_tokens"] = tokenized_text
            output["text"] = caption

        if enable_metadata:
            metadata_file = item["json"]
            metadata = metadata_file.decode("utf-8")
            output["metadata"] = metadata
        return output

    transformed_dataset = filtered_dataset.map(
        preprocess_dataset, handler=wds.handlers.warn_and_continue)
    return transformed_dataset


def dataset_to_dataloader(dataset, batch_size, num_prepro_workers, input_format, pin_memory):
    """Create a pytorch dataloader from a dataset"""

    def collate_fn(batch):
        batch = list(filter(lambda x: x is not None, batch))
        return default_collate(batch)

    data = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_prepro_workers,
        pin_memory=pin_memory,
        prefetch_factor=None,
        collate_fn=collate_fn if input_format == "files" else None,
    )
    return data


class WebdatasetReader:
    """WebdatasetReader is a reader that reads samples from a webdataset"""

    def __init__(
        self,
        preprocess,
        input_dataset_folder,
        batch_size,
        num_prepro_workers,
        enable_text=True,
        enable_image=True,
        enable_metadata=False,
        wds_image_key="jpg",
        wds_caption_key="txt",
        cache_path=None,
        pin_memory=True
    ):
        self.batch_size = batch_size

        all_files = os.listdir(input_dataset_folder)
        tar_files = list(filter(lambda x: x.endswith(".tar"), all_files))
        print(f'reading from {len(tar_files)} tar files')
        full_paths = list(map(lambda x: os.path.join(
            input_dataset_folder, x), tar_files))

        dataset = create_webdataset(
            urls=full_paths,
            image_transform=preprocess,
            enable_text=enable_text,
            enable_image=enable_image,
            image_key=wds_image_key,
            caption_key=wds_caption_key,
            enable_metadata=enable_metadata,
            cache_path=cache_path,
        )
        self.dataloader = dataset_to_dataloader(
            dataset, batch_size, num_prepro_workers, "webdataset", pin_memory)

    def __iter__(self):
        for batch in self.dataloader:
            yield batch


class LAIONDataModule(pl.LightningDataModule):
    def __init__(
        self,
        input_dataset_folder,
        batch_size,
        num_prepro_workers,
        enable_text=True,
        enable_image=True,
        enable_metadata=False,
        wds_image_key="jpg",
        wds_caption_key="txt",
        preprocess='RN50',
        cache_path=None,
        pin_memory=True
    ):
        super().__init__()
        self.input_dataset_folder = input_dataset_folder
        self.batch_size = batch_size
        self.num_prepro_workers = num_prepro_workers
        self.enable_text = enable_text
        self.enable_image = enable_image
        self.enable_metadata = enable_metadata
        self.wds_image_key = wds_image_key
        self.wds_caption_key = wds_caption_key
        self.cache_path = cache_path
        self.pin_memory = pin_memory

        _, self.preprocess = clip.load(name=preprocess)

    def setup(self, stage):
        if stage == "fit" or stage is None:
            self.train_dataset = WebdatasetReader(
                self.preprocess,
                self.input_dataset_folder,
                self.batch_size,
                self.num_prepro_workers,
                self.enable_text,
                self.enable_image,
                self.enable_metadata,
                self.wds_image_key,
                self.wds_caption_key,
                self.cache_path,
                self.pin_memory
            )

        if stage == 'validate':
            self.val_dataset = WebdatasetReader(
                self.preprocess,
                self.input_dataset_folder,
                self.batch_size,
                self.num_prepro_workers,
                self.enable_text,
                self.enable_image,
                self.enable_metadata,
                self.wds_image_key,
                self.wds_caption_key,
                self.cache_path,
                self.pin_memory
            )

    def train_dataloader(self):
        return self.train_dataset

    def val_dataloader(self):
        return self.val_dataset


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    input_dataset = "dataset/laion400m-data"
    batch_size = 5
    num_prepro_workers = 0
    _, preprocess = clip.load(clip.available_models()[0])

    output_partition_count = 2
    actual_values = []
    reader = WebdatasetReader(
        preprocess,
        input_dataset,
        batch_size,
        num_prepro_workers,
        enable_text=True,
        enable_image=True,
        enable_metadata=True,
    )

    for batch in reader:
        break  # placeholder
