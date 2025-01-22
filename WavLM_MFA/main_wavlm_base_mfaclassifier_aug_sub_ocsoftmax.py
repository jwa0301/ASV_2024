"""
Main script that trains, validates, and evaluates
various models including AASIST.

AASIST
Copyright (c) 2021-present NAVER Corp.
MIT license
"""
import argparse
import json
import os
import sys

from importlib import import_module
from pathlib import Path
from shutil import copy
from typing import Dict, List, Union
from tqdm import tqdm
#from models.Wav2vec2_base import Model as w2v2_model
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchcontrib.optim import SWA

from data_utils_ocsoftmax import (Dataset_ASVspoof2024_train_aug, Dataset_ASVspoof2024_eval,
                        Dataset_ASVspoof2024_train_aug2,
                        Dataset_ASVspoof2019_devNeval, genSpoof_list)
from evaluation_ocsoftmax import calculate_tDCF_EER, calculate_EER
from utils import create_optimizer, seed_worker, set_seed, str_to_bool, OCSoftmax

import warnings
warnings.filterwarnings('ignore')
import logging
import sys
import zipfile

print("----------------------------------------------------")
logging.disable(logging.INFO) # disable INFO and DEBUG logging everywhere
logging.disable(logging.WARNING)

def to_MB(a):
    return a/1024.0/1024.0
def main(args: argparse.Namespace) -> None:
    """
    Main function.
    Trains, validates, and evaluates the ASVspoof detection model.
    """
    # load experiment configurations
    with open(args.config, "r") as f_json:
        config = json.loads(f_json.read())
    model_config = config["model_config"]
    optim_config = config["optim_config"]
    optim_config["epochs"] = config["num_epochs"]
    track = config["track"]
    assert track in ["t1", "t2"], "Invalid track given"
    if "eval_all_best" not in config:
        config["eval_all_best"] = "True"
    if "freq_aug" not in config:
        config["freq_aug"] = "False"
        
    # config["model_config"]["total_transformer_layers"] = tl
    # config["model_config"]["n_frz_layers"] = fl
    device = int(config["model_config"]["device"])
    
    # print('Total Transformer Layer : ',tl)
    # print('Freeze up to N Layer : ',fl)
    if model_config['noise_rir'] != 0 :
        print('*'*30,'Add NOISE & RIR Aug','*'*30)
#    print('LoRA Config : ',config["model_config"]["lora_config"])
    
    # make experiment reproducible
    set_seed(args.seed, config)

    # define database related paths
    output_dir = Path(args.output_dir)
    prefix_2019 = "ASVspoof2024.{}".format(track)
    database_path = Path(config["database_path"])
    dev_trial_path = (database_path /
                      "ASVspoof5.dev.metadata.txt")
    dev_samp_trial_path = (database_path /
                      "ASVspoof5.dev.metadata_sampling.txt")

    # define model related paths
    model_tag = "{}_{}_ep{}_bs{}_lr{}".format(
        track,
        os.path.splitext(os.path.basename(args.config))[0],
        config["num_epochs"], config["batch_size"],config["optim_config"]["base_lr"])
        # ,config["model_config"]["total_transformer_layers"],config["model_config"]["n_frz_layers"])
#        ,config["model_config"]["lora_config"]["r"])
    if args.comment:
        model_tag = model_tag + "_{}".format(args.comment)
    model_tag = output_dir / model_tag
    print('Results save at : ',model_tag)
    model_save_path = model_tag / "weights"
    eval_score_path = model_tag / config["eval_output"]
    writer = SummaryWriter(model_tag)
    os.makedirs(model_save_path, exist_ok=True)
    copy(args.config, model_tag / "config.conf")

    # set device
    device = torch.device('cuda:{}'.format(device) if torch.cuda.is_available() else 'cpu')
    print("Device: {}".format(device))
    if device == "cpu":
        raise ValueError("GPU not detected!")

    # define model architecture
    model = get_model(model_config, device)
    loss_model = OCSoftmax(1536, r_real=0.9, r_fake=0.3, alpha=20).to(device)
    model.print_trainable_parameters()
    print(f"After model to device: {to_MB(torch.cuda.memory_allocated(device)):.2f}MB")
    print(f"After model to device cache: {to_MB(torch.cuda.memory_reserved(device)):.2f}MB")
    # define dataloaders
#    trn_loader, dev_loader, eval_loader = get_loader(
#        database_path, args.seed, config)
    trn_loader, dev_loader, dev_samp_loader, eval_loader = get_loader(
        database_path, args.seed, config)

    # evaluates pretrained model and exit script
    if args.eval:
        eval_score_folder = './submission/'+config["model_path"].split('/')[-3]
        os.makedirs(eval_score_folder, exist_ok=True)
        # eval_score_path = eval_score_folder + '/' + config["eval_output"]
        model.load_state_dict(
            torch.load(config["model_path"], map_location=device))
        
        state_dict = torch.load(config["loss_model_path"], map_location=device)
        loss_model.load_state_dict(state_dict)
        
        print("Model loaded : {}".format(config["model_path"]))
        print("loss_model loaded : {}".format(config["loss_model_path"]))
        print("Start evaluation...")
        produce_evaluation_score_file(eval_loader, model, loss_model, device, eval_score_path)
        print("DONE.")
#        calculate_tDCF_EER(cm_scores_file=eval_score_path,
#                           asv_score_file=database_path /
#                           config["asv_score_path"],
#                           output_file=model_tag / "t-DCF_EER.txt")
#        print("DONE.")
#        eval_eer, eval_tdcf = calculate_tDCF_EER(
#            cm_scores_file=eval_score_path,
#            asv_score_file=database_path / config["asv_score_path"],
#            output_file=model_tag/"loaded_model_t-DCF_EER.txt")
        sys.exit(0)

    # get optimizer and scheduler
    optim_config["steps_per_epoch"] = len(trn_loader)
    optimizer, scheduler = create_optimizer(model.parameters(), optim_config)
    optimizer_swa = SWA(optimizer)

    # ocsoftmax optimizer, schedular 
    ocsoftmax = OCSoftmax(1536, r_real=0.9, r_fake=0.3, alpha=20).to(device) # 맨 앞은 enc_dim
    # ocsoftmax_optimizer = torch.optim.SGD(ocsoftmax.parameters(), lr=optim_config['base_lr'])
    ocsoftmax_optimizer = torch.optim.Adam(ocsoftmax.parameters(), lr=optim_config['base_lr'])
    ocsoftmax_scheduler = torch.optim.lr_scheduler.StepLR(ocsoftmax_optimizer, step_size=30, gamma=0.1)
    
    best_dev_eer = 50
    best_eval_eer = 100.
    best_dev_tdcf = 0.05
    best_eval_tdcf = 1.
    n_swa_update = 0  # number of snapshots of model to use in SWA
    f_log = open(model_tag / "metric_log.txt", "a")
    f_log.write("=" * 5 + "\n")

    # make directory for metric logging
    metric_path = model_tag / "metrics"
    os.makedirs(metric_path, exist_ok=True)

    # Training
    for epoch in range(config["num_epochs"]):
        print("Start training epoch{:03d}".format(epoch))
        running_loss = train_epoch(trn_loader, model, ocsoftmax, optimizer, ocsoftmax_optimizer,
                                   device, scheduler, ocsoftmax_scheduler, config)
        
        if epoch < config['num_epochs'] -5:
            bona_cm, spoof_cm = produce_evaluation_file(dev_samp_loader, model, ocsoftmax, device,
                                metric_path/"dev_score.txt", dev_samp_trial_path)
        else :
            bona_cm, spoof_cm = produce_evaluation_file(dev_loader, model, ocsoftmax, device,
                                metric_path/"dev_score.txt", dev_trial_path)
        dev_eer = calculate_EER(
            cm_scores_file=metric_path/"dev_score.txt",
            output_file=metric_path/"dev_t-DCF_EER_{}epo.txt".format(epoch),
            bona_cm=bona_cm,
            spoof_cm=spoof_cm,
            printout=False)
        print("DONE.\nLoss:{:.5f}, dev_eer: {:.3f}".format(
            running_loss, dev_eer))
                      
        writer.add_scalar("loss", running_loss, epoch)
        writer.add_scalar("dev_eer", dev_eer, epoch)
#        writer.add_scalar("dev_tdcf", dev_tdcf, epoch)

#        best_dev_tdcf = min(dev_tdcf, best_dev_tdcf)
        if best_dev_eer >= dev_eer:
            print("best model find at epoch", epoch)
            best_dev_eer = dev_eer
            torch.save(model.state_dict(),
                       model_save_path / "best.pth")#epoch_{}_{:03.3f}.pth".format(epoch, dev_eer))##############3
            torch.save(ocsoftmax.state_dict(), 
                      model_save_path / "ocsoftmax_best.pth")
            
            f_log = open(model_tag / "metric_log.txt", "a")
            f_log.write("=" * 5 + "\n")
            f_log.write("Epoch: {}, EER: {:.3f}\n".format(epoch,best_dev_eer))

            f_log.close()
'''
            # do evaluation whenever best model is renewed
            if str_to_bool(config["eval_all_best"]):
                produce_evaluation_file(eval_loader, model, device,
                                        eval_score_path, eval_trial_path)
                eval_eer, eval_tdcf = calculate_tDCF_EER(
                    cm_scores_file=eval_score_path,
                    asv_score_file=database_path / config["asv_score_path"],
                    output_file=metric_path /
                    "t-DCF_EER_{:03d}epo.txt".format(epoch))

                log_text = "epoch{:03d}, ".format(epoch)
                if eval_eer < best_eval_eer:
                    log_text += "best eer, {:.4f}%".format(eval_eer)
                    best_eval_eer = eval_eer
                    torch.save(model.state_dict(),
                               model_save_path / "best.pth")
                if eval_tdcf < best_eval_tdcf:
                    log_text += "best tdcf, {:.4f}".format(eval_tdcf)
                    best_eval_tdcf = eval_tdcf
                    
                if len(log_text) > 0:
                    print(log_text)
                    f_log.write(log_text + "\n")

            print("Saving epoch {} for swa".format(epoch))
            optimizer_swa.update_swa()
            n_swa_update += 1
        writer.add_scalar("best_dev_eer", best_dev_eer, epoch)
        writer.add_scalar("best_dev_tdcf", best_dev_tdcf, epoch)

    print("Start final evaluation")
    epoch += 1
    if n_swa_update > 0:
        optimizer_swa.swap_swa_sgd()
        optimizer_swa.bn_update(trn_loader, model, device=device)
    produce_evaluation_file(eval_loader, model, device, eval_score_path,
                            eval_trial_path)
    eval_eer, eval_tdcf = calculate_tDCF_EER(cm_scores_file=eval_score_path,
                                             asv_score_file=database_path /
                                             config["asv_score_path"],
                                             output_file=model_tag / "t-DCF_EER.txt")
                      
    f_log = open(model_tag / "metric_log.txt", "a")
    f_log.write("=" * 5 + "\n")
    f_log.write("EER: {:.3f}, min t-DCF: {:.5f}\n".format(eval_eer, eval_tdcf))
    f_log.write(model.print_trainable_parameters(return_option=True))
    f_log.write(f"\nAfter model to device: {to_MB(torch.cuda.memory_allocated(device)):.2f}MB\n")
    f_log.write(f"\nAfter model to device cache: {to_MB(torch.cuda.memory_reserved(device)):.2f}MB\n")
    
    f_log.close()
    
    f_log = open(model_tag / "t-DCF_EER.txt", "a")
    f_log.write("=" * 30 + "\n")
    f_log.write(model.print_trainable_parameters(return_option=True))
    f_log.write(f"\nAfter model to device: {to_MB(torch.cuda.memory_allocated(device)):.2f}MB\n")
    f_log.write(f"\nAfter model to device cache: {to_MB(torch.cuda.memory_reserved(device)):.2f}MB\n")
    
    f_log.close()

    torch.save(model.state_dict(),
               model_save_path / "swa.pth")

    if eval_eer <= best_eval_eer:
        best_eval_eer = eval_eer
        torch.save(model.state_dict(),
                   model_save_path / "best.pth")
    if eval_tdcf <= best_eval_tdcf:                    ### tdcf ㄱㅣㅈㅜㄴ best model
        best_eval_tdcf = eval_tdcf
        
    print("Exp FIN. EER: {:.3f}, min t-DCF: {:.5f}".format(
        best_eval_eer, best_eval_tdcf))
'''

def get_model(model_config: Dict, device: torch.device):
    """Define DNN model architecture"""
    module = import_module("models.{}".format(model_config["architecture"]))
    _model = getattr(module, "Model")
    model = _model(model_config).to(device)
    nb_params = sum([param.view(-1).size()[0] for param in model.parameters()])
    print("no. model params:{}".format(nb_params))

    return model


def get_loader(
        database_path: str,
        seed: int,
        config: dict) -> List[torch.utils.data.DataLoader]:
    """Make PyTorch DataLoaders for train / developement / evaluation"""
    track = config["track"]
    noise_rir = config['model_config']['noise_rir']

    trn_database_path = database_path / "flac_T/"
    dev_database_path = database_path / "flac_D/"
    eval_database_path = database_path / "flac_E_prog/"

    trn_list_path = (database_path /
                     "/Data3/ASV2024/ASVspoof5.train.metadata_13aug_4000.txt")
    dev_trial_path = (database_path /
                      "ASVspoof5.dev.metadata.txt")
    dev_samp_trial_path = (database_path /
                      "ASVspoof5.dev.metadata_sampling.txt")
#    eval_trial_path = (
#        database_path /
#        "ASVspoof2019_{}_cm_protocols/{}.cm.eval.trl.txt".format(
#            track, prefix_2019))

    d_label_trn, file_train = genSpoof_list(dir_meta=trn_list_path,
                                            is_train=True,
                                            is_eval=False)
    print("no. training files:", len(file_train))
    if noise_rir == 0 :
        train_set = Dataset_ASVspoof2024_train_aug(list_IDs=file_train,
                                               labels=d_label_trn,
                                               base_dir=trn_database_path)
    else : 
        train_set = Dataset_ASVspoof2024_train_aug2(list_IDs=file_train,
                                               labels=d_label_trn,
                                               base_dir=trn_database_path,
                                               add = noise_rir)
    gen = torch.Generator()
    gen.manual_seed(seed)
    trn_loader = DataLoader(train_set,
                            batch_size=config["batch_size"],
                            shuffle=True,
                            drop_last=True,
                            pin_memory=True,
                            worker_init_fn=seed_worker,
                            generator=gen)

    d_label_dev, file_dev = genSpoof_list(dir_meta=dev_trial_path,
                                is_train=False,
                                is_eval=False)
    print("no. validation files:", len(file_dev))

    dev_set = Dataset_ASVspoof2019_devNeval(list_IDs=file_dev,
                                            labels=d_label_dev,
                                            base_dir=dev_database_path)
    dev_loader = DataLoader(dev_set,
                            batch_size=config["batch_size"],
                            shuffle=False,
                            drop_last=False,
                            pin_memory=True)
    d_label_dev_semp, file_dev_samp = genSpoof_list(dir_meta=dev_samp_trial_path,
                                is_train=False,
                                is_eval=False)
    print("no. sampling validation files:", len(file_dev_samp))

    dev_samp_set = Dataset_ASVspoof2019_devNeval(list_IDs=file_dev_samp,
                                                 labels=d_label_dev_semp,
                                            base_dir=dev_database_path)
    dev_samp_loader = DataLoader(dev_samp_set,
                            batch_size=config["batch_size"],
                            shuffle=False,
                            drop_last=False,
                            pin_memory=True)
    
    file_eval = genSpoof_list(dir_meta=eval_database_path,
                                is_train=False,
                                is_eval=True)
    print("no. evaluation files:", len(file_eval))

    eval_set = Dataset_ASVspoof2024_eval(list_IDs=file_eval,
                                            base_dir=eval_database_path)
    eval_loader = DataLoader(eval_set,
                            batch_size=config["batch_size"],
                            shuffle=False,
                            drop_last=False,
                            pin_memory=True)

    return trn_loader, dev_loader, dev_samp_loader, eval_loader


def produce_evaluation_file(
    data_loader: DataLoader,
    model,
    ocsoftmax,
    device: torch.device,
    save_path: str,
    trial_path: str) -> None:
    """Perform evaluation and save the score to a file"""
    model.eval()
    with open(trial_path, "r") as f_trl:
        trial_lines = f_trl.readlines()
    fname_list = []
    score_list = []
    idx_loader, score_loader = [], [] # add line
    
    for batch_x, utt_id, batch_y in tqdm(data_loader):
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device) # add line
        with torch.no_grad():
            feats, batch_out = model(batch_x)
            batch_score = (batch_out[:, 1]).data.cpu().numpy().ravel()

            ocsoftmaxloss, score = ocsoftmax(feats, batch_y) # add line
            idx_loader.append(batch_y)   # add line
            score_loader.append(score)   # add line 
            
        # add outputs
        fname_list.extend(utt_id)
        score_list.extend(score.tolist())
        
    scores = torch.cat(score_loader, 0).data.cpu().numpy()  # add line
    labels = torch.cat(idx_loader, 0).data.cpu().numpy()  # add line
    
    assert len(trial_lines) == len(fname_list) == len(score_list)
    with open(save_path, "w") as fh:
        for fn, sco, trl in zip(fname_list, score_list, trial_lines):
            _, utt_id, _,_, src, key = trl.strip().split(' ')
            assert fn == utt_id
            fh.write("{} {} {} {}\n".format(utt_id, src, key, sco))
    print("Scores saved to {}".format(save_path))

    return scores[labels == 1], scores[labels == 0]  # add line
    
def produce_evaluation_score_file(
    data_loader: DataLoader,
    model,
    loss_model,
    device: torch.device,
    save_path: str) -> None:
    """Perform evaluation and save the score to a file"""
    model.eval()
    loss_model.eval()
    
    # sub_path = save_path.replace('score.tsv','submission.zip')
    sub_path = str(save_path).replace('score.tsv','submission.zip')
    fname_list = []
    score_list = []
    for batch_x, utt_id in tqdm(data_loader):
        batch_x = batch_x.to(device)
        with torch.no_grad():
            feats, batch_out = model(batch_x)
            _, batch_score = loss_model(feats) # add
            batch_score =  batch_score.data.cpu().numpy().ravel()
            # batch_score = (batch_out[:, 1]).data.cpu().numpy().ravel()
        # add outputs
        fname_list.extend(utt_id)
        score_list.extend(batch_score.tolist())

    assert len(fname_list) == len(score_list)
    with open(save_path, "w") as fh:
        fh.write("filename\tcm-score\n")
        for fn, sco in zip(fname_list, score_list):
            if fn.startswith("E_A"):  # 
                continue
            fh.write("{}\t{}\n".format(fn, sco))
    fh.close()
    zip_file = zipfile.ZipFile(sub_path, "w")
    # zip_file.write(save_path, compress_type=zipfile.ZIP_DEFLATED)
    zip_file.write(save_path, arcname=save_path.name, compress_type=zipfile.ZIP_DEFLATED)
    zip_file.close()
    
    
    print("Scores saved to {}".format(save_path))
    print("Zip File saved to {}".format(sub_path))


def train_epoch(
    trn_loader: DataLoader,
    model,
    ocsoftmax,
    optim: Union[torch.optim.SGD, torch.optim.Adam],
    ocsoftmax_optimizer: Union[torch.optim.SGD, torch.optim.Adam],
    device: torch.device,
    scheduler: torch.optim.lr_scheduler,
    ocsoftmax_scheduler,
    config: argparse.Namespace):
    """Train the model for one epoch"""
    running_loss = 0
    num_total = 0.0
    ii = 0
    model.train()

    # set objective (Loss) functions
    weight = torch.FloatTensor([0.1, 0.9]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight)
    for batch_x, batch_y in tqdm(trn_loader):
        batch_size = batch_x.size(0)
        num_total += batch_size
        ii += 1
        batch_x = batch_x.to(device)
        batch_y = batch_y.view(-1).type(torch.int64).to(device)
        # _, batch_out = model(batch_x, Freq_aug=str_to_bool(config["freq_aug"]))
        feats, batch_out = model(batch_x, Freq_aug=str_to_bool(config["freq_aug"]))
        # batch_loss = criterion(batch_out, batch_y)

        batch_ocsoftmaxloss, _ = ocsoftmax(feats, batch_y)
        batch_loss = batch_ocsoftmaxloss
        
        running_loss += batch_loss.item() * batch_size
        optim.zero_grad()
        ocsoftmax_optimizer.zero_grad()
        batch_loss.backward()
        optim.step()
        ocsoftmax_optimizer.step()
        ocsoftmax_scheduler.step()
        
        if config["optim_config"]["scheduler"] in ["cosine", "keras_decay"]:
            scheduler.step()
        elif scheduler is None:
            pass
        else:
            raise ValueError("scheduler error, got:{}".format(scheduler))

    running_loss /= num_total
    return running_loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ASVspoof detection system")
    parser.add_argument("--config",
                        dest="config",
                        type=str,
                        help="configuration file",
                        default="Data3/dh24/asv2024_dh2/config/wavlm_mfaclassifier_ocsoftmax.conf")
                   #     required=True)
    parser.add_argument(
        "--output_dir",
        dest="output_dir",
        type=str,
        help="output directory for results",
        default="Data3/dh24/asv2024_dh2/exp_optim_model",
    )
    parser.add_argument("--seed",
                        type=int,
                        default=1234,
                        help="random seed (default: 1234)")
    parser.add_argument(
                        "--eval",
                        action="store_true",
                        help="when this flag is given, evaluates given model and exit")
    parser.add_argument("--comment",
                        type=str,
                        default=None,
                        help="comment to describe the saved model")
    parser.add_argument("--eval_model_weights",
                        type=str,
                        default=None,
                        help="directory to the model weight file (can be also given in the config file)")


main(parser.parse_args())
    # total_transformer_layer_space = [9]
    # freeze_layer_space = [3]
    # print('Total_Transformer_Layer_Space : ',total_transformer_layer_space)
    # print('Freeze_Layer_Space : ',freeze_layer_space)
    
    # for tl in total_transformer_layer_space :
    #     for fl in freeze_layer_space :
    #         if tl > fl :
    #             main(parser.parse_args(),tl,fl)
    #         else :
    #             continue