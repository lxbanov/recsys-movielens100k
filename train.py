import shutil
import argparse

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers.wandb import WandbLogger
import torch
from torch.utils.data import DataLoader
from scripts.model import RecModelLightning
from scripts.utils import Movie100DatasetNoText


parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default='data/raw')
parser.add_argument('--embedding_dim', type=int, default=512)
parser.add_argument('--joint_dim', type=int, default=512)
parser.add_argument('--dropout', type=float, default=0.2)
parser.add_argument('--hidden_dim', type=int, default=512)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--weight_decay', type=float, default=1e-6)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--num_workers', type=int, default=1)
parser.add_argument('--max_epochs', type=int, default=10)
parser.add_argument('--checkpoint_path', type=str, default='models/interim')
parser.add_argument('--seed', type=int, default=705)
parser.add_argument('--val_check_interval', type=float, default=0.25)

args = parser.parse_args()

train_dataset = Movie100DatasetNoText(
    folder_path='data/interim/train'
)

val_dataset = Movie100DatasetNoText(
    folder_path='data/interim/val'
)

lit = RecModelLightning(
    user_dim=train_dataset.user_shape,
    item_dim=train_dataset.item_shape,
    embedding_dim=args.embedding_dim,
    joint_dim=args.joint_dim,
    dropout=args.dropout,
    hidden_dim=args.hidden_dim,
    lr=args.lr,
    weight_decay=args.weight_decay,
)

# Settings for checkpointing and early stopping
checkpoint_callback = ModelCheckpoint(
    monitor='val_loss',
    dirpath=args.checkpoint_path,
    filename='model-{epoch:02d}-{val_loss:.2f}',
    save_top_k=3,
    mode='min',
)

early_stop_callback = EarlyStopping(
    monitor='val_loss',
    patience=3,
    mode='min',
)

# Log to wandb to track training
wandb_logger = WandbLogger(
    project='recsys',
    log_model=True,
    save_dir=args.checkpoint_path,
)

# Train the model
trainer = Trainer(
    max_epochs=args.max_epochs,
    callbacks=[checkpoint_callback, early_stop_callback],
    logger=wandb_logger,
    accelerator='auto',
    val_check_interval=args.val_check_interval,
)

trainer_loader = DataLoader(
    train_dataset,
    batch_size=args.batch_size,
    shuffle=True,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=args.batch_size,
    shuffle=False,
)

torch.manual_seed(args.seed)

trainer.fit(
    lit,
    train_dataloaders=trainer_loader,
    val_dataloaders=val_loader,
)

trainer.validate(
    dataloaders=val_loader,
)

# Save model
shutil.copy(checkpoint_callback.best_model_path, 'models/best.ckpt')
