#!/usr/bin/env python3
import time
from multiprocessing import cpu_count
from typing import Union, NamedTuple

import torch
import torch.backends.cudnn
import numpy as np
from torch import nn, optim
from torch.nn import functional as F
import torchvision.datasets
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from model import LMCNet, MCNet, MLMCNet, TSCNN
from dataset import UrbanSound8KDataset

import argparse
from pathlib import Path

torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser(
    description="Train a TSCNN on UrbanSound8KDataset",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument("--log-dir", default=Path("TSCNN_logs"), type=Path)
parser.add_argument("--learning-rate", default=1e-3, type=float, help="Learning rate")
parser.add_argument(
    "--batch-size",
    default=32,
    type=int,
    help="Number of images within each mini-batch",
)
parser.add_argument(
    "--epochs",
    default=50,
    type=int,
    help="Number of epochs (passes through the entire dataset) to train for",
)
parser.add_argument(
    "--val-frequency",
    default=2,
    type=int,
    help="How frequently to test the model on the validation set in number of epochs",
)
parser.add_argument(
    "--log-frequency",
    default=10,
    type=int,
    help="How frequently to save logs to tensorboard in number of steps",
)
parser.add_argument(
    "--print-frequency",
    default=10,
    type=int,
    help="How frequently to print progress to the command line in number of steps",
)
parser.add_argument(
    "-j",
    "--worker-count",
    default=cpu_count(),
    type=int,
    help="Number of worker processes used to load data.",
)
parser.add_argument(
    "--dropout",
    default=0.5,
    type=float,
    help="Dropout",
)


if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")


def main(args):
    LMC_train_loader = torch.utils.data.DataLoader(
        UrbanSound8KDataset('UrbanSound8K_train.pkl', "LMC"),
        shuffle=True,
        batch_size=args.batch_size,
        pin_memory=True,
        num_workers=args.worker_count,
    )

    MC_train_loader = torch.utils.data.DataLoader(
        UrbanSound8KDataset('UrbanSound8K_train.pkl', "MC"),
        shuffle=True,
        batch_size=args.batch_size,
        pin_memory=True,
        num_workers=args.worker_count,
    )


    LMC_test_loader = torch.utils.data.DataLoader(
        UrbanSound8KDataset('UrbanSound8K_test.pkl', "LMC"),
        shuffle=False,
        batch_size=args.batch_size,
        num_workers=args.worker_count,
        pin_memory=True,
    )

    MC_test_loader = torch.utils.data.DataLoader(
        UrbanSound8KDataset('UrbanSound8K_test.pkl', "MC"),
        shuffle=True,
        batch_size=args.batch_size,
        pin_memory=True,
        num_workers=args.worker_count,
    )

    # The length of LMC & MC dataset are equal
    train_dataset_length=len(LMC_train_loader.dataset)
    test_dataset_length=len(LMC_test_loader.dataset)


    LMC_model = LMCNet(height=85, width=41, channels=1, class_count=10,dropout=args.dropout)
    MC_model = MCNet(height=85, width=41, channels=1, class_count=10,dropout=args.dropout)

    ## TASK 8: Redefine the criterion to be softmax cross entropy
    criterion = nn.CrossEntropyLoss()

    ## TASK 11: Define the optimizer
    LMC_optimizer = optim.SGD(LMC_model.parameters(), lr=args.learning_rate,momentum=0.9)
    MC_optimizer = optim.SGD(MC_model.parameters(), lr=args.learning_rate,momentum=0.9)

    log_dir = get_summary_writer_log_dir(args)
    print(f"Writing logs to {log_dir}")
    summary_writer = SummaryWriter(
            str(log_dir),
            flush_secs=5
    )

    LMC_trainer = Trainer(LMC_model, LMC_train_loader, LMC_test_loader, criterion, LMC_optimizer, DEVICE)
    MC_trainer = Trainer(MC_model, MC_train_loader, MC_test_loader, criterion, MC_optimizer, DEVICE)

    LMC_trainer.train(
        args.epochs,
        args.val_frequency,
        print_frequency=args.print_frequency,
        log_frequency=args.log_frequency,
    )

    MC_trainer.train(
        args.epochs,
        args.val_frequency,
        print_frequency=args.print_frequency,
        log_frequency=args.log_frequency,
    )

    LMC_train_records=LMC_trainer.train_records
    MC_train_records=MC_trainer.train_records

    assert len(LMC_train_records)==len(MC_train_records)

    for i in range(0,len(LMC_train_records)):
        LMC_step ,LMC_epoch, LMC_logits, LMC_labels=LMC_train_records[i]
        MC_step, MC_epoch, MC_logits, MC_labels=MC_train_records[i]

        assert LMC_step==MC_step
        step=LMC_step

        assert LMC_epoch==MC_epoch
        epoch=LMC_epoch

        assert np.all(LMC_labels==MC_labels)
        labels=LMC_labels

        LMC_logits=torch.from_numpy(LMC_logits)
        MC_logits=torch.from_numpy(MC_logits)

        labels=torch.from_numpy(labels)
        logits = torch.mean(torch.stack((LMC_logits,MC_logits), dim=2),dim=2)
        preds = logits.argmax(-1)

        accuracy=compute_accuracy(labels,preds)
        loss = criterion(logits,labels)

        if ((step + 1) % args.log_frequency) == 0:
            log_metrics(summary_writer,"train",epoch,accuracy,loss,step)
        if ((step + 1) % args.print_frequency) == 0:
            print_metrics(step,epoch,accuracy,loss)

    LMC_test_records=LMC_trainer.test_records
    MC_test_records=MC_trainer.test_records

    assert len(LMC_test_records)==len(MC_test_records)

    for i in range(0,len(LMC_test_records)):
        LMC_step, LMC_epoch, LMC_logits, LMC_labels=LMC_train_records[i]
        MC_step, MC_epoch, MC_logits, MC_labels=MC_train_records[i]

        assert LMC_step==MC_step
        step=LMC_step

        assert LMC_epoch==MC_epoch
        epoch=LMC_epoch

        assert np.all(LMC_labels==MC_labels)
        labels=LMC_labels

        LMC_logits=torch.from_numpy(LMC_logits)
        MC_logits=torch.from_numpy(MC_logits)

        labels=torch.from_numpy(labels)
        logits = torch.mean(torch.stack((LMC_logits,MC_logits), dim=2),dim=2)
        preds = logits.argmax(-1)

        accuracy=compute_accuracy(labels,preds)
        loss = criterion(logits,labels)

        labels=MC_labels
        compute_class_accuracy(labels,preds.numpy())

        log_metrics(summary_writer,"test",epoch,accuracy,loss,step)
        print_metrics(step,epoch,accuracy,loss)

    summary_writer.close()



class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        optimizer: Optimizer,
        device: torch.device,
    ):
        self.model = model.to(device)
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.step = 0
        self.train_records= []
        self.test_records=[]

    def train(
        self,
        epochs: int,
        val_frequency: int,
        print_frequency: int = 20,
        log_frequency: int = 5,
        start_epoch: int = 0
    ):
        self.model.train()
        for epoch in range(start_epoch, epochs):
            self.model.train()
            data_load_start_time = time.time()
            for i, (batch, labels, filenames, labelnames) in enumerate(self.train_loader):
                batch = batch.to(self.device)
                labels = labels.to(self.device)

                logits = self.model.forward(batch)

                loss = self.criterion(logits,labels)

                loss.backward()

                self.optimizer.step()
                self.optimizer.zero_grad()

                logits_array=logits.clone().detach().cpu().numpy()
                labels_array=labels.clone().detach().cpu().numpy()

                # with torch.no_grad():
                #     preds = logits.argmax(-1)
                #     accuracy = compute_accuracy(labels, preds)

                self.train_records.append((self.step, epoch, logits_array, labels_array))
                # if ((self.step + 1) % log_frequency) == 0:
                #     self.log_metrics(epoch, accuracy, loss, data_load_time, step_time)
                # if ((self.step + 1) % print_frequency) == 0:
                #     self.print_metrics(epoch, accuracy, loss, data_load_time, step_time)

                self.step += 1
                data_load_start_time = time.time()

            # self.summary_writer.add_scalar("epoch", epoch, self.step)
            if ((epoch + 1) % val_frequency) == 0:
                self.validate(epoch)
                # self.validate() will put the model in validation mode,
                # so we have to switch back to train mode afterwards
                self.model.train()

    def validate(self,epoch):
        dict = {}
        # results = {"preds": [], "labels": []}
        self.model.eval()

        # No need to track gradients for validation, we're not optimizing.
        with torch.no_grad():
            for i, (batch, labels, filenames, labelnames) in enumerate(self.val_loader):
                batch = batch.to(self.device)
                labels = labels.to(self.device)
                logits = self.model(batch)

                # loss = self.criterion(logits, labels)
                # total_loss += loss.item()

                logits_array=logits.cpu().numpy()
                labels_array=labels.cpu().numpy()
                batch_size = len(filenames)
                for j in range(0,batch_size):
                    filename=filenames[j]
                    if filename in dict:
                        count = dict[filename]['count']
                        dict[filename]['average']=(count*dict[filename]['average']+logits_array[j])/(count+1)
                        dict[filename]['count']+=1
                    else:
                        dict[filename]={}
                        dict[filename]['average']=logits_array[j]
                        dict[filename]['label']=labels_array[j]
                        dict[filename]['count']=1

        labels_list=[dict[k]['label'] for k,v in dict.items()]
        logits_list=[dict[k]['average'] for k,v in dict.items()]
        labels_array=np.hstack(labels_list)
        logits_array=np.vstack(logits_list)
        self.test_records.append((self.step, epoch, logits_array, labels_array))
        # labels=torch.from_numpy(labels_array).to(self.device)
        # logits=torch.from_numpy(logits_array).to(self.device)
        #
        # loss = self.criterion(logits, labels)
        #
        # # preds = logits.argmax(dim=-1).cpu().numpy()
        # preds = np.argmax(logits_array,axis=-1)
        # results["preds"].extend(list(preds))
        # results["labels"].extend(list(labels_array))
        #
        # accuracy = compute_accuracy(
        #     np.array(results["labels"]), np.array(results["preds"])
        # )
        # compute_class_accuracy(np.array(results["labels"]), np.array(results["preds"]))

        # self.summary_writer.add_scalars(
        #         "accuracy",
        #         {"test": accuracy},
        #         self.step
        # )
        # self.summary_writer.add_scalars(
        #         "loss",
        #         {"test": float(loss.item())},
        #         self.step
        # )
        # print(f"validation loss: {average_loss:.5f}, accuracy: {accuracy * 100:2.2f}")


def print_metrics(step, epoch, accuracy, loss):
    print(
            f"epoch: [{epoch}], "
            f"step: [{step}], "
            f"batch loss: {loss:.5f}, "
            f"batch accuracy: {accuracy * 100:2.2f}, "
    )

def log_metrics(summary_writer, type,epoch, accuracy, loss,step):
    summary_writer.add_scalar("epoch", epoch, step)
    summary_writer.add_scalars(
            "accuracy",
            {type: accuracy},
            step
    )
    summary_writer.add_scalars(
            "loss",
            {type: float(loss.item())},
            step
    )

def compute_accuracy(labels: Union[torch.Tensor, np.ndarray], preds: Union[torch.Tensor, np.ndarray]) -> float:
    """
    Args:
        labels: ``(batch_size, class_count)`` tensor or array containing example labels
        preds: ``(batch_size, class_count)`` tensor or array containing model prediction
    """
    assert len(labels) == len(preds)
    return float((labels == preds).sum()) / len(labels)


def compute_class_accuracy(labels: np.ndarray, preds: np.ndarray):
    assert len(labels) == len(preds)
    targets=torch.from_numpy(labels).float().to(DEVICE)
    predictions=torch.from_numpy(preds).float().to(DEVICE)
    for c in range(0,10):
        mask=lambda x:x==c
        index_of_targets_with_class_c=torch.nonzero(mask(targets))
        index_of_preds_with_class_c=torch.nonzero(mask(predictions))
        number_of_class_c_targets=len(index_of_targets_with_class_c)
        if number_of_class_c_targets==0:
            class_accuracy=0.0
        else:
            count=0
            for i in index_of_targets_with_class_c:
                if i in index_of_preds_with_class_c:
                    count+=1
                else:
                    continue
            class_accuracy=count/number_of_class_c_targets

        txt = "Accuracy for class "+str(c)+") is: "+str(class_accuracy*100)
        print(txt)

def get_summary_writer_log_dir(args: argparse.Namespace) -> str:
    """Get a unique directory that hasn't been logged to before for use with a TB
    SummaryWriter.

    Args:
        args: CLI Arguments

    Returns:
        Subdirectory of log_dir with unique subdirectory name to prevent multiple runs
        from getting logged to the same TB log directory (which you can't easily
        untangle in TB).
    """
    tb_log_dir_prefix = (
                         f"CNN_bn_"
                         f"dropout={args.dropout}_"
                         f"bs={args.batch_size}_"
                         f"lr={args.learning_rate}_"
                         f"momentum=0.9_"
                         f"run_"
                         )
    i = 0
    while i < 1000:
        tb_log_dir = args.log_dir / (tb_log_dir_prefix + str(i))
        if not tb_log_dir.exists():
            return str(tb_log_dir)
        i += 1
    return str(tb_log_dir)



if __name__ == "__main__":
    main(parser.parse_args())
