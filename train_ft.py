import argparse
import torch
from PIL import Image
from torchvision import transforms
import torchvision.models as models
from torch.nn import functional as F
import os
import numpy as np

from my_dassl.utils import setup_logger, set_random_seed, collect_env_info
from my_dassl.config import get_cfg_default
from my_dassl.engine import build_trainer
from my_dassl.data.datasets import Datum
from my_dassl.utils import read_image

# import datasets.oxford_pets
import datasets.restrict_imagenet
import datasets.svhn

import trainers.ftvgg16
import trainers.ftresnet50

import pdb
import matplotlib.pyplot as plt

def print_args(args, cfg):
    print("***************")
    print("** Arguments **")
    print("***************")
    optkeys = list(args.__dict__.keys())
    optkeys.sort()
    for key in optkeys:
        print("{}: {}".format(key, args.__dict__[key]))
    print("************")
    print("** Config **")
    print("************")
    print(cfg)


def reset_cfg(cfg, args):
    if args.root:
        cfg.DATASET.ROOT = args.root
    
    if args.eval_only: cfg.eval_only = 1
    else:              cfg.eval_only = 0

    if args.output_dir:
        cfg.OUTPUT_DIR = args.output_dir
    
    if args.resume:
        cfg.RESUME = args.resume

    if args.seed:
        cfg.SEED = args.seed

    if args.wb_method_name != 'no':
        cfg.WB_METHOD_NAME = args.wb_method_name
    
    if args.use_wandb: cfg.use_wandb = 1
    else:              cfg.use_wandb = 0

    cfg.EVAL_MODE = 'best'

    if args.source_domains:
        cfg.DATASET.SOURCE_DOMAINS = args.source_domains

    if args.target_domains:
        cfg.DATASET.TARGET_DOMAINS = args.target_domains

    if args.transforms:
        cfg.INPUT.TRANSFORMS = args.transforms

    if args.trainer:
        cfg.TRAINER.NAME = args.trainer

    if args.backbone:
        cfg.MODEL.BACKBONE.NAME = args.backbone

    if args.head:
        cfg.MODEL.HEAD.NAME = args.head

    if args.adv_folder:
        cfg.TEST.ADVERSARIAL_FOLDER = args.adv_folder

    if args.reverse_test:
        cfg.TEST.REVERSE = args.reverse_test

    if args.prompt_attack:
        cfg.ATTACK.PROMPT = args.prompt_attack

    if args.inference_only:
        cfg.TEST.INFERENCE = args.inference_only


def extend_cfg(cfg):
    """
    Add new config variables.

    E.g.
        from yacs.config import CfgNode as CN
        cfg.TRAINER.MY_MODEL = CN()
        cfg.TRAINER.MY_MODEL.PARAM_A = 1.
        cfg.TRAINER.MY_MODEL.PARAM_B = 0.5
        cfg.TRAINER.MY_MODEL.PARAM_C = False
    """
    from yacs.config import CfgNode as CN

    #! DATASET CONFIG
    cfg.DATASET.SUBSAMPLE_CLASSES = "all"  # all, base or new
    
    cfg.DATASET.LOCMNIST = CN()
    cfg.DATASET.LOCMNIST.R_SIZE = 1
    cfg.DATASET.LOCMNIST.F_SIZE = 4

    cfg.DATASET.COLOUR_BIASED_MNIST = CN()
    cfg.DATASET.COLOUR_BIASED_MNIST.TRAIN_RHO = 0.8
    cfg.DATASET.COLOUR_BIASED_MNIST.TEST_RHO = 0.2
    cfg.DATASET.COLOUR_BIASED_MNIST.TRAIN_N_CONFUSING_LABELS = 9
    cfg.DATASET.COLOUR_BIASED_MNIST.TEST_N_CONFUSING_LABELS = 9
    cfg.DATASET.COLOUR_BIASED_MNIST.USE_TEST_AS_VAL = True
    cfg.DATASET.COLOUR_BIASED_MNIST.RANDOMIZE = True if args.randomize else False

    # Calculate per-class accuracy
    cfg.TEST.PER_CLASS_RESULT = False

    # Calculate adversarial attack accuracy
    cfg.TEST.ADVERSARIAL_ATTACK = ''

    # Skip some classes for adversarial attack accuracy
    cfg.TEST.START_CLASS = 0

    #! Bahng et al. Visual Prompting (VP)
    cfg.TRAINER.VPWB = CN()
    cfg.TRAINER.VPWB.PREC = "amp"  # fp16, fp32, amp
    cfg.TRAINER.VPWB.METHOD = 'padding' # 'padding', 'fixed_patch', 'random_patch'
    cfg.TRAINER.VPWB.IMAGE_SIZE = 224
    cfg.TRAINER.VPWB.PROMPT_SIZE = 30

    #! Visual Prompting (VP) with SPSA
    cfg.TRAINER.VPOUR = CN()
    cfg.TRAINER.VPOUR.METHOD = 'padding' 
    cfg.TRAINER.VPOUR.IMAGE_SIZE = 224
    cfg.TRAINER.VPOUR.PROMPT_SIZE = 30
    cfg.TRAINER.VPOUR.SPSA_PARAMS = [0.0,0.001,40.0,0.6,0.1]
    cfg.TRAINER.VPOUR.OPT_TYPE = "spsa-gc"
    cfg.TRAINER.VPOUR.MOMS = 0.9
    cfg.TRAINER.VPOUR.SP_AVG = 5

    #! BlackVIP
    cfg.TRAINER.BLACKVIP = CN()
    cfg.TRAINER.BLACKVIP.METHOD = 'coordinator'
    cfg.TRAINER.BLACKVIP.PT_BACKBONE = 'vit-mae-base' # vit-base / vit-mae-base
    cfg.TRAINER.BLACKVIP.SRC_DIM = 1568 # 784 / 1568 / 3136 #? => only for pre-trained Enc
    cfg.TRAINER.BLACKVIP.E_OUT_DIM = 0 # 64 / 128 / 256 #? => only for scratch Enc
    cfg.TRAINER.BLACKVIP.SPSA_PARAMS = [0.0,0.001,40.0,0.6,0.1]
    cfg.TRAINER.BLACKVIP.OPT_TYPE = "spsa-gc" # [spsa, spsa-gc, naive]
    cfg.TRAINER.BLACKVIP.MOMS = 0.9 # first moment scale.
    cfg.TRAINER.BLACKVIP.SP_AVG = 5 # grad estimates averaging steps
    cfg.TRAINER.BLACKVIP.P_EPS = 1.0 # prompt scale

    #! Black-Box Adversarial Reprogramming (BAR)
    cfg.TRAINER.BAR = CN()
    cfg.TRAINER.BAR.METHOD = 'reprogramming'
    cfg.TRAINER.BAR.LRS = [0.01, 0.0001]
    cfg.TRAINER.BAR.FRAME_SIZE = 224
    cfg.TRAINER.BAR.SMOOTH = 0.01
    cfg.TRAINER.BAR.SIMGA = 1.0
    cfg.TRAINER.BAR.SP_AVG = 5
    cfg.TRAINER.BAR.FOCAL_G = 2.0

    #! Full Fine Tune / Linear Probe 
    cfg.TRAINER.FTCLIP = CN()
    cfg.TRAINER.FTCLIP.METHOD = 'ft'       # 'ft', 'lp'

    #! CoOp, CoCoOp
    cfg.TRAINER.COOP = CN()
    cfg.TRAINER.COOP.N_CTX = 16  # number of context vectors
    cfg.TRAINER.COOP.CSC = False  # class-specific context
    cfg.TRAINER.COOP.CTX_INIT = ""  # initialization words
    cfg.TRAINER.COOP.PREC = "fp16"  # fp16, fp32, amp
    cfg.TRAINER.COOP.CLASS_TOKEN_POSITION = "end"  # 'middle' or 'end' or 'front'

    cfg.TRAINER.COCOOP = CN()
    cfg.TRAINER.COCOOP.N_CTX = 16  # number of context vectors
    cfg.TRAINER.COCOOP.CTX_INIT = ""  # initialization words
    cfg.TRAINER.COCOOP.PREC = "fp16"  # fp16, fp32, amp


def setup_cfg(args):
    cfg = get_cfg_default()
    extend_cfg(cfg)

    # 1. From the dataset config file
    if args.dataset_config_file:
        cfg.merge_from_file(args.dataset_config_file)

    # 2. From the method config file
    if args.config_file:
        cfg.merge_from_file(args.config_file)

    # 3. From input arguments
    reset_cfg(cfg, args)

    # 4. From optional input arguments
    cfg.merge_from_list(args.opts)

    cfg.freeze()
    return cfg

def reverse_preprocessing(image, pixel_mean, pixel_std):
    # Get the mean and std values from the configuration
    # Reverse the normalization
    reverse_mean = pixel_mean
    reverse_std = pixel_std
    for c in range(3):
        image[c] *= reverse_std[c]
        image[c] += reverse_mean[c]

    # Clip values to be in the valid range
    image[image > 1] = 1
    image[image < 0] = 0

    # Scale back to [0, 255]
    image = np.round(image * 255)

    # Transpose from CHW to HWC
    image = np.uint8(image).transpose(1, 2, 0)

    return image

def predict_image_class(adv_trainer, image_path, model=None):

    # Create a datum with the image path
    datum = Datum(impath=image_path, label=-1, domain=-1)
    classnames = adv_trainer.dm.dataset.classnames
    image = read_image(datum.impath) # Read the image
    image = adv_trainer.dm.tfm_test(image) # Apply the necessary transforms
    image = image.to(adv_trainer.device).unsqueeze(0)  # add batch dimension

    # Set the model to evaluation mode
    adv_trainer.set_model_mode("eval")

    # Perform inference
    if model is None: 
        with torch.no_grad():
            prompted_image = adv_trainer.get_prompted_img(image.cuda())
            logits = adv_trainer.model_inference(prompted_image.cuda()) 
    else: 
        model.eval()
        logits = torch.nn.Softmax(dim=1).forward(model.forward(image.cuda()))

    # Get the predicted class
    _, predicted_class = torch.max(logits, 1)
    predicted_class_name = classnames[predicted_class.item()] # Get the class name using the predicted class index

    return image, prompted_image, predicted_class_name, predicted_class

def visualize_attack(adv_trainer, original_image, perturbed_image, label_image, cfg, model=None, attacked_img=None):
    # Set the model to evaluation mode
    adv_trainer.set_model_mode("eval")

    # Perform inference on the perturbed images
    if model is None: 
        with torch.no_grad(): perturbed_logits = adv_trainer.model_inference(perturbed_image.cuda())
    else: 
        model.eval()
        perturbed_logits = torch.nn.Softmax(dim=1).forward(model.forward(perturbed_image.cuda()))

    # Get the predicted classes
    _, perturbed_predicted_class = torch.max(perturbed_logits, 1)
    perturbed_predicted_class_name = adv_trainer.dm.dataset.classnames[perturbed_predicted_class.item()] # Get the class names for attacked images

    # Rescale the pixel values for display
    if cfg.TEST.ADVERSARIAL_ATTACK == 'ZOsignSGD' and torch.max(perturbed_image) == 0.5:
        perturbed_image_disp = (perturbed_image + 0.5)* (torch.max(original_image) - torch.min(original_image)) + torch.min(original_image)

    elif cfg.ATTACK.PROMPT: 
        perturbed_image_disp = (perturbed_image)* (torch.max(original_image) - torch.min(original_image)) + torch.min(original_image)

    else: 
        perturbed_image_disp = (attacked_img)* (torch.max(original_image) - torch.min(original_image)) + torch.min(original_image)

    # Reverse the preprocessing on the original and perturbed images
    original_image_disp = reverse_preprocessing(original_image[0].cpu().numpy(), cfg.INPUT.PIXEL_MEAN, cfg.INPUT.PIXEL_STD)
    perturbed_image_disp = reverse_preprocessing(perturbed_image_disp[0].cpu().numpy(), cfg.INPUT.PIXEL_MEAN, cfg.INPUT.PIXEL_STD)

    # Plot the original and perturbed images with their predicted labels
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(original_image_disp)
    ax[0].set_title(f'Original Image\nPredicted Label: {label_image}')
    ax[1].imshow(perturbed_image_disp)
    ax[1].set_title(f'Attacked Image with {cfg.TEST.ADVERSARIAL_ATTACK}\nPredicted Label: {perturbed_predicted_class_name}')

    plt.show()

def visualize_attack_folder(adv_trainer, folder_path, target_folder, cfg, attacker, model=None):
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    for class_folder in os.listdir(folder_path):
        class_folder_path = os.path.join(folder_path, class_folder)
        if not os.path.isdir(class_folder_path):
            continue  # Skip non-directory files

        for filename in os.listdir(class_folder_path):
            image_path = os.path.join(class_folder_path, filename)
            image, prompted_image, label_img, tensor_label = predict_image_class(adv_trainer, image_path, model=model)

            if cfg.ATTACK.PROMPT:
                perturbed_data = attacker.batch_attack(prompted_image, tensor_label)
            else: 
                attacked_img = attacker.batch_attack(image, tensor_label)
                perturbed_data = adv_trainer.get_prompted_img(attacked_img)

            # Generate visualization and save it
            save_visualization(adv_trainer, image, perturbed_data, label_img, cfg, target_folder, filename, model=model, attacked_img=attacked_img)

def save_visualization(adv_trainer, original_image, perturbed_image, label_image, cfg, target_folder, filename, model=None, attacked_img=None):
    # Set the model to evaluation mode
    adv_trainer.set_model_mode("eval")

    # Perform inference on the perturbed images
    if model is None:
        with torch.no_grad():
            perturbed_logits = adv_trainer.model_inference(perturbed_image.cuda())
    else:
        model.eval()
        perturbed_logits = torch.nn.Softmax(dim=1).forward(model.forward(perturbed_image.cuda()))

    # Get the predicted classes
    _, perturbed_predicted_class = torch.max(perturbed_logits, 1)
    perturbed_predicted_class_name = adv_trainer.dm.dataset.classnames[perturbed_predicted_class.item()]

    # Rescale the pixel values for display
    if cfg.TEST.ADVERSARIAL_ATTACK == 'ZOsignSGD' and torch.max(perturbed_image) == 0.5:
        perturbed_image_disp = (perturbed_image + 0.5) * (torch.max(original_image) - torch.min(original_image)) + torch.min(original_image)
    elif cfg.ATTACK.PROMPT:
        perturbed_image_disp = (perturbed_image) * (torch.max(original_image) - torch.min(original_image)) + torch.min(original_image)
    else:
        perturbed_image_disp = (attacked_img) * (torch.max(original_image) - torch.min(original_image)) + torch.min(original_image)

    # Reverse the preprocessing on the original and perturbed images
    original_image_disp = reverse_preprocessing(original_image[0].cpu().numpy(), cfg.INPUT.PIXEL_MEAN, cfg.INPUT.PIXEL_STD)
    perturbed_image_disp = reverse_preprocessing(perturbed_image_disp[0].cpu().numpy(), cfg.INPUT.PIXEL_MEAN, cfg.INPUT.PIXEL_STD)

    # Plot the original and perturbed images with their predicted labels
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(original_image_disp)
    ax[0].set_title(f'Original Image\nPredicted Label: {label_image}')
    ax[1].imshow(perturbed_image_disp)
    ax[1].set_title(f'Attacked Image with {cfg.TEST.ADVERSARIAL_ATTACK}\nPredicted Label: {perturbed_predicted_class_name}')

    # Save the figure
    save_path = os.path.join(target_folder, filename.replace('.jpg', '_visualization.jpg'))
    fig.savefig(save_path)
    plt.close(fig)  # Close the figure to free memory


def main(args):
    cfg = setup_cfg(args)
    model=None
    if cfg.SEED >= 0:
        print("Setting fixed seed: {}".format(cfg.SEED))
        set_random_seed(cfg.SEED)
    setup_logger(cfg.OUTPUT_DIR)

    if torch.cuda.is_available() and cfg.USE_CUDA:
        torch.backends.cudnn.benchmark = True

    # print_args(args, cfg)
    print("Collecting env info ...")
    print("** System info **\n{}\n".format(collect_env_info()))

    trainer = build_trainer(cfg)

    if cfg.DATASET.SUBSAMPLE_CLASSES != "all": #! base-to-new generalization setting
        # base class
        if (not args.no_train) and (not args.eval_only):
            trainer.train()
            trainer = build_trainer(cfg)  #! re-build for selecting test-model
        else:
            pass
        
        # new class
        cfg.DATASET.defrost()
        cfg.DATASET.SUBSAMPLE_CLASSES = "new"
        cfg.DATASET.freeze()

        trainer = build_trainer(cfg)
        trainer.load_model(trainer.output_dir, epoch=cfg.OPTIM.MAX_EPOCH)
        trainer.test()
            
    else: #! normal setting (use all classes)
        if (not args.no_train) and (not args.eval_only) and (not args.inference_only):
            if cfg.TRAIN.ADVERSARIAL_ATTACK != '':
                trainer.train_adv()
            else:
                trainer.train()
            trainer = build_trainer(cfg)
            trainer.load_model(trainer.output_dir, epoch=cfg.OPTIM.MAX_EPOCH)

            adv_trainer = AdvTrainer(cfg, trainer, model)
            if cfg.TRAIN.ADVERSARIAL_ATTACK != '' and cfg.MODEL.TORCHVISION == "":
                adv_trainer.test_attack()
            elif cfg.TRAIN.ADVERSARIAL_ATTACK != '' and cfg.MODEL.TORCHVISION != "":
                adv_trainer.torchvision_test_att()
            else:
                trainer.test()
                
        elif args.inference_only and (not args.no_train) and (not args.eval_only):

            if cfg.TRAINER.NAME == 'ZeroshotCLIP' or cfg.TRAINER.NAME == 'ZeroshotCLIP2': trainer.load_model(args.model_dir, epoch=args.load_epoch)
            else: trainer.load_model(trainer.output_dir, epoch=args.load_epoch)

            adv_trainer = AdvTrainer(cfg, trainer, model)
            attacker = adv_trainer.init_adv_attack(trainer, model)

            if cfg.TEST.INFER_FOLDER:
                # If INFER_FOLDER is true, perform visualization on a folder of images
                visualize_attack_folder(adv_trainer, args.image_folder, args.target_folder, cfg, attacker, model=model)
            else:
                # Regular visualization for a single image
                image, prompted_image, label_img, tensor_label = predict_image_class(adv_trainer, args.image_path, model=model)
                attacker = adv_trainer.init_adv_attack(trainer, model)

                if args.advimage_path == "" and cfg.ATTACK.PROMPT:
                    perturbed_data = attacker.batch_attack(prompted_image, tensor_label)
                    visualize_attack(adv_trainer, prompted_image, perturbed_data, label_img, cfg, model=model)
                elif args.advimage_path == "":
                    attacked_img = attacker.batch_attack(image, tensor_label)
                    perturbed_data = adv_trainer.get_prompted_img(attacked_img)
                    visualize_attack(adv_trainer, image, perturbed_data, label_img, cfg, model=model, attacked_img=attacked_img)
                else:
                    perturbed_data = transforms.ToTensor()(Image.open(args.advimage_path)).to(trainer.device).unsqueeze(0)
                    visualize_attack(adv_trainer, image, perturbed_data, label_img, cfg, model=model)
            
        else: # eval_only

            if cfg.TRAINER.NAME == 'ZeroshotCLIP' or cfg.TRAINER.NAME == 'ZeroshotCLIP2':
                trainer.load_model(args.model_dir, epoch=args.load_epoch)
            else:
                trainer.load_model(trainer.output_dir, epoch=args.load_epoch)

            # import pdb; pdb.set_trace()
            adv_trainer = AdvTrainer(cfg, trainer, model)

            if cfg.TEST.ADVERSARIAL_ATTACK != '' and (cfg.TEST.ADVERSARIAL_FOLDER): 
                adv_trainer.test_normal() 
            elif cfg.TEST.ADVERSARIAL_ATTACK != '' and cfg.TEST.NO_BLACKVIP: 
                adv_trainer.test_attack() 
            elif cfg.TEST.ADVERSARIAL_ATTACK != '': 
                adv_trainer.test_attack_blackvip()       
            else: 
                trainer.test()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="/media/icl1/Data/ravialdy/VLP_Research_Topic/BlackVIP/data", help="path to dataset")
    parser.add_argument("--output-dir", type=str, default="", help="output directory")
    parser.add_argument("--resume",type=str,default="",help="checkpoint directory (from which the training resumes)")
    parser.add_argument("--seed", type=int, default=1, help="only positive value enables a fixed seed")
    parser.add_argument("--source-domains", type=str, nargs="+", help="source domains for DA/DG")
    parser.add_argument("--target-domains", type=str, nargs="+", help="target domains for DA/DG")
    parser.add_argument("--transforms", type=str, nargs="+", help="data augmentation methods")
    parser.add_argument("--config-file", type=str, default="", help="path to config file")
    parser.add_argument("--dataset-config-file",type=str,default="",help="path to config file for dataset setup",)
    parser.add_argument("--trainer", type=str, default="", help="name of trainer")
    parser.add_argument("--backbone", type=str, default="", help="name of CNN backbone")
    parser.add_argument("--head", type=str, default="", help="name of head")
    parser.add_argument("--eval-only", action="store_true", help="evaluation only")
    parser.add_argument("--inference-only", action="store_true", help="inference only")
    parser.add_argument("--model-dir",type=str,default="",help="load model from this directory for eval-only mode",)
    parser.add_argument("--load-epoch", type=int, help="load model weights at this epoch for evaluation")
    parser.add_argument("--no-train", action="store_true", help="do not call trainer.train()")
    parser.add_argument("--image-path", type=str, default="", help="image directory for inference")
    parser.add_argument("--advimage-path", type=str, default="", help="adversarial image directory for inference")
    parser.add_argument("--adv-folder", action="store_true", default="", help="adversarial images for evaluation")
    parser.add_argument("--reverse-test", type=str, default="", help="reverse the order of the model for evaluation")
    parser.add_argument("--prompt-attack", default=False, action="store_true", help="whether to attack in the prompt or not")
    parser.add_argument("--image-folder", type=str, default="", help="path to the folder containing images for inference")
    parser.add_argument("--target-folder", type=str, default="", help="path to the target folder where visualizations will be saved")
    #! extension
    parser.add_argument('--use_wandb', default=False, action="store_true", help='whether to use wandb')
    parser.add_argument('--wb_name', type=str, default='test', help='wandb project name')
    parser.add_argument('--wb_method_name', type=str, default='no')
    parser.add_argument('--randomize', type=int, default=1)
    parser.add_argument("opts",default=None,nargs=argparse.REMAINDER,help="modify config options using the command-line",)
    
    args = parser.parse_args()
    main(args)