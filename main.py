#-*- coding:utf-8 -*-
import os, sys
import time
import pprint
import shutil
import numpy as np
import torch
import torch.nn as nn
import argparse
import yaml
import matplotlib.pyplot as plt
from model.net import Net
from addict import Dict


def parse_args():
    parser = argparse.ArgumentParser(description="train a network for ***")
    parser.add_argument("config", type=str, help="path of a config file")
    parser.add_argument("--date", type=str, default="")
    args = parser.parse_args()
    return args


def sec2str(sec):
    if sec < 60:
        return "elapsed: {:02d}s".format(int(sec))
    elif sec < 3600:
        min = int(sec / 60)
        sec = int(sec - min * 60)
        return "elapsed: {:02d}m{:02d}s".format(min, sec)
    elif sec < 24 * 3600:
        min = int(sec / 60)
        hr = int(min / 60)
        sec = int(sec - min * 60)
        min = int(min - hr * 60)
        return "elapsed: {:02d}h{:02d}m{:02d}s".format(hr, min, sec)
    elif sec < 365 * 24 * 3600:
        min = int(sec / 60)
        hr = int(min / 60)
        dy = int(hr / 24)
        sec = int(sec - min * 60)
        min = int(min - hr * 60)
        hr = int(hr - dy * 24)
        return "elapsed: {:02d} days, {:02d}h{:02d}m{:02d}s".format(dy, hr, min, sec)


def train(train_dataloader, val_dataloader, net, criterion, optimizer, lr_scheduler, date, CONFIG):
    print("start train and validation")

    result_dir = os.path.join(CONFIG.result_dir, date)
    if os.path.exists(result_dir):
        shutil.rmtree(result_dir)
    os.makedirs(result_dir, exist_ok=True)
    
    checkpoint_dir = os.path.join(CONFIG.checkpoint_dir)
    if not os.path.exists(checkpoint_dir):
        shutil.rmtree(checkpoint_dir)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    train_loss_list = []
    val_loss_list = []
    best_loss = 1e8
    for epoch in range(CONFIG.epoch_num):
        start = time.time()
        train_loss = 0
        val_loss = 0
        net.train()
        print("epoch", epoch + 1)
        for i, data in enumerate(train_dataloader):
            x = data["x"].to(device)
            label = data["label"].to(device)

            optimizer.zero_grad()
            output = net(x)
            
            loss = criterion(output, label)
            train_loss += loss.item()
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), CONFIG.get("clip", 0.5))
            optimizer.step()
            # break
        lr_scheduler.step(loss)
        avg_train_loss = train_loss / len(train_dataloader)
        
        train_time = sec2str(time.time() - start)
        print("train", train_time)
        # break

        start = time.time()
        net.eval()
        with torch.no_grad():
            for i, data in enumerate(val_dataloader):
                x = data["x"].to(device)
                label = data["label"].to(device)

                output = net(x)

                loss = criterion(output, label)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_dataloader)
        torch.save(net.state_dict(), os.path.join(checkpoint_dir, "checkpoint.pth"))       
        if avg_val_loss <= best_loss:
            print("save parameters")
            torch.save(net.state_dict(), os.path.join(result_dir, "checkpoint.pth"))
            best_loss = avg_val_loss

        val_time = sec2str(time.time() - start)
        print("validation", val_time)
        print(
            "Epoch [{}/{}], train_loss: {loss:.4f}, val_loss: {val_loss:.4f}".format(
                epoch + 1,
                CONFIG.epoch_num,
                loss=avg_train_loss,
                val_loss=avg_val_loss
            )
        )
        train_loss_list.append(avg_train_loss)
        val_loss_list.append(avg_val_loss)
        
        plt.figure()
        plt.plot(train_loss_list, label="train")
        plt.plot(val_loss_list, label="val")
        plt.yscale("log")
        plt.legend()
        plt.savefig(os.path.join(result_dir, "loss.png"))
        plt.close()
        # break


def test(test_dataloader, net, date, CONFIG):
    print("start test")
    start = time.time()
    result_path = os.path.join(CONFIG.result_dir, date, "checkpoint.pth")
    net.load_state_dict(torch.load(result_path))
    net.eval()
    with torch.no_grad():
        total = 0
        test_acc = 0
        for i, data in enumerate(test_dataloader):
            x = data["x"].to(device)
            label = data["label"].to(device)

            output = net(text)
    print("精度: {} %".format(100))
    print("test", sec2str(time.time() - start))


def main(date):
    # argparser
    args = parse_args()
    if args.date != "":
        date = args.date
    CONFIG = Dict(yaml.safe_load(open(args.config)))
    # pprint.pprint(CONFIG)

    # cpu or gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print(device)
    if device == "cpu":
        print("You have to use GPUs because training CNN is computationally expensive.")
        sys.exit(1)
    random.seed(CONFIG.seed)
    np.random.seed(CONFIG.seed)
    torch.manual_seed(CONFIG.seed)
    torch.autograd.set_detect_anomaly(True)
    torch.cuda.manual_seed_all(CONFIG.seed)
    torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.enabled = False
    
    # loading dataset
    print("Loading ...")
    start = time.time()

    train_dataset, test_dataset = Dataset(**CONFIG)
    test_dataset, val_dataset = test_dataset.split()
    train_dataloader = torch.utils.data.DataLoader(
        dataset = train_dataset,
        batch_size = CONFIG.batch_size,
        shuffle = True,
        drop_last = True,
        num_workers = CONFIG.num_workers,
        # collate_fn = collate_fn,
        # worker_init_fn = worker_init_fn
    )
    val_dataloader = torch.utils.data.DataLoader(
        dataset = val_dataset,
        batch_size = CONFIG.batch_size,
        shuffle = False,
        drop_last = True,
        num_workers = CONFIG.num_workers,
        # collate_fn = collate_fn,
        # worker_init_fn = worker_init_fn
    )
    test_dataloader = torch.utils.data.DataLoader(
        dataset = test_dataset,
        batch_size = CONFIG.batch_size,
        shuffle = False,
        drop_last = True,
        num_workers = CONFIG.num_workers,
        # collate_fn = collate_fn,
        # worker_init_fn = worker_init_fn
    )
    print("Loading time", sec2str(time.time() - start))

    print(CONFIG.model)
    net = Net(word_embeddings, CONFIG)
    if torch.cuda.device_count() > 1 and CONFIG.rnn == "Transformer":
        device_ids = range(torch.cuda.device_count())
        net = nn.DataParallel(net, device_ids=device_ids)
    net = net.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        net.parameters(),
        lr=float(CONFIG.learning_rate),
        weight_decay=float(CONFIG.weight_decay)
    )
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=float(CONFIG.factor),
        verbose=True,
        min_lr=float(CONFIG.min_learning_rate),
    )

    train(train_dataloader, val_dataloader, net, criterion, optimizer, lr_scheduler, date, CONFIG)
    test(test_dataloader, net, date, CONFIG)
    print("finished", sec2str(time.time() - start))


if __name__ == "__main__":
    start_main = time.time()
    start_now = datetime.datetime.now()
    date = start_now.strftime("%Y-%m-%d")
    print(start_now.strftime("%Y/%m/%d %H:%M:%S"))

    main(date)

    end_main = sec2str(time.time() - start_main)
    end_now = datetime.datetime.now()
    print(
        "Finished main.py! | {} | {}".format(
        end_main, end_now.strftime("%Y/%m/%d %H:%M:%S"))
    )
    print("="*70)
