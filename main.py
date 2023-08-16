import os
import yaml
import argparse
import pytorch_lightning as pl
import clip
from model import stage_one, stage_two
from experiment import DistillExperiment
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from dataset.dataset import LAIONDataModule
from pytorch_lightning.strategies import DDPStrategy


parser = argparse.ArgumentParser()
parser.add_argument('--config', '-c',
                    dest="filename",
                    metavar='FILE',
                    help='path to the config file',
                    default='configs/stage_one.yaml')

args = parser.parse_args()
with open(args.filename, 'r') as file:
    try:
        config = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        print(exc)


tb_logger = TensorBoardLogger(save_dir=config['logging_params']['save_dir'],
                              name=config['model_params']['name'],)

# For reproducibility
pl.seed_everything(config['exp_params']['manual_seed'], True)

if config['model_params']['stage'] == 1:
    model = stage_one.DistillModel(**config['model_params'])
elif config['model_params']['stage'] == 2:
    model = stage_two.DistillModel(**config['model_params'])
else:
    raise ValueError("Invalid stage number")

experiment = DistillExperiment(model,
                               config['exp_params'])

_, preprocess = clip.load(name=clip.available_models()[0])
data = LAIONDataModule(**config["data_params"],
                       pin_memory=(config['trainer_params']['accelerator'] == 'gpu'))

data.setup('fit')
data.setup('validate')
runner = Trainer(logger=tb_logger,
                 callbacks=[
                     LearningRateMonitor(),
                     ModelCheckpoint(save_top_k=2,
                                     dirpath=os.path.join(tb_logger.log_dir, "checkpoints"),
                                     monitor="val_loss",
                                     save_last=True),
                 ],
                 **config['trainer_params'])


print(f"======= Training {config['model_params']['name']} =======")
runner.fit(experiment, datamodule=data)
