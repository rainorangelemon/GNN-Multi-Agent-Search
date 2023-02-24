import os
import torch
import numpy as np
import random
import importlib
from PIL import Image

import logging
from logging import Formatter
from logging.handlers import RotatingFileHandler
from typing import Any, Callable, Optional, Union, Dict, Generator

import json
from easydict import EasyDict
from pprint import pprint
import datetime
import time
from pathlib import Path

log = logging.getLogger(__name__)


class DotDict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def save_gif(imgs, filename="play.gif"):
    # Setup the 4 dimensional array
    a_frames = []
    for img in imgs:
        a_frames.append(np.asarray(img))
    a = np.stack(a_frames)

    ims = [Image.fromarray(a_frame) for a_frame in a]
    ims[0].save(filename, save_all=True, append_images=ims[1:], loop=0, duration=50)    
    

def create_dirs(dirs):
    '''
    Create directories in the systm
    Args:
        dirs: a list of directories to create if these directories are not found

    Returns:

    '''
    try:
        for dir_ in dirs:
            if not os.path.exists(dir_):
                os.makedirs(dir_)
    except Exception as err:
        logging.getLogger("Dirs Creator").info("Creating directories error: {0}".format(err))
        exit(-1)

def _select_seed_randomly(min_seed_value: int = 0, max_seed_value: int = 255) -> int:
    return random.randint(min_seed_value, max_seed_value)


def seed_everything(seed: Optional[int] = None, workers: bool = False) -> int:
    """Function that sets seed for pseudo-random number generators in: pytorch, numpy, python.random In addition,
    sets the following environment variables:

    - `PL_GLOBAL_SEED`: will be passed to spawned subprocesses (e.g. ddp_spawn backend).
    - `PL_SEED_WORKERS`: (optional) is set to 1 if ``workers=True``.

    Args:
        seed: the integer value seed for global random state in Lightning.
            If `None`, will read seed from `PL_GLOBAL_SEED` env variable
            or select it randomly.
        workers: if set to ``True``, will properly configure all dataloaders passed to the
            Trainer with a ``worker_init_fn``. If the user already provides such a function
            for their dataloaders, setting this argument will have no influence. See also:
            :func:`~pytorch_lightning.utilities.seed.pl_worker_init_function`.
    """
    max_seed_value = np.iinfo(np.uint32).max
    min_seed_value = np.iinfo(np.uint32).min

    if seed is None:
        env_seed = os.environ.get("PL_GLOBAL_SEED")
        if env_seed is None:
            seed = _select_seed_randomly(min_seed_value, max_seed_value)
        else:
            try:
                seed = int(env_seed)
            except ValueError:
                seed = _select_seed_randomly(min_seed_value, max_seed_value)
    elif not isinstance(seed, int):
        seed = int(seed)

    if not (min_seed_value <= seed <= max_seed_value):
        seed = _select_seed_randomly(min_seed_value, max_seed_value)

    # using `log.info` instead of `rank_zero_info`,
    # so users can verify the seed is properly set in distributed training.
    log.info(f"Global seed set to {seed}")
    os.environ["PL_GLOBAL_SEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    os.environ["PL_SEED_WORKERS"] = f"{int(workers)}"

    return seed


def setup_logging(log_dir):
    log_file_format = "[%(levelname)s] - %(asctime)s - %(name)s - : %(message)s in %(pathname)s:%(lineno)d"
    log_console_format = "[%(levelname)s]: %(message)s"

    # Main logger
    main_logger = logging.getLogger()
    main_logger.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(Formatter(log_console_format))

    exp_file_handler = RotatingFileHandler('{}exp_debug.log'.format(log_dir), maxBytes=10**6, backupCount=5)
    exp_file_handler.setLevel(logging.DEBUG)
    exp_file_handler.setFormatter(Formatter(log_file_format))

    exp_errors_file_handler = RotatingFileHandler('{}exp_error.log'.format(log_dir), maxBytes=10**6, backupCount=5)
    exp_errors_file_handler.setLevel(logging.WARNING)
    exp_errors_file_handler.setFormatter(Formatter(log_file_format))

    main_logger.addHandler(console_handler)
    main_logger.addHandler(exp_file_handler)
    main_logger.addHandler(exp_errors_file_handler)


def get_config_from_json(json_file):
    """
    Get the config from a json file
    :param json_file: the path of the config file
    :return: config(namespace), config(dictionary)
    """

    # parse the configurations from the config json file provided
    with open(json_file, 'r') as config_file:
        try:
            config_dict = json.load(config_file)
            # EasyDict allows to access dict values as attributes (works recursively).
            config = EasyDict(config_dict)
            return config, config_dict
        except ValueError:
            print("INVALID JSON file format.. Please provide a good json file")
            exit(-1)


def print_cuda_statistics():
    logger = logging.getLogger("Cuda Statistics")
    import sys
    from subprocess import call
    import torch
    logger.info('__Python VERSION:  {}'.format(sys.version))
    logger.info('__pyTorch VERSION:  {}'.format(torch.__version__))
    logger.info('__CUDA VERSION')
    # call(["nvcc", "--version"])
    logger.info('__CUDNN VERSION:  {}'.format(torch.backends.cudnn.version()))
    logger.info('__Number CUDA Devices:  {}'.format(torch.cuda.device_count()))
    logger.info('__Devices')
    call(["nvidia-smi", "--format=csv",
          "--query-gpu=index,name,driver_version,memory.total,memory.used,memory.free"])
    logger.info('Active CUDA Device: GPU {}'.format(torch.cuda.current_device()))
    logger.info('Available devices  {}'.format(torch.cuda.device_count()))
    logger.info('Current cuda device  {}'.format(torch.cuda.current_device()))


def process_config(config):
    """
    Get the json file
    Processing it with EasyDict to be accessible as attributes
    then editing the path of the experiments folder
    creating some important directories in the experiment folder
    Then model_setup the logging in the whole program
    Then return the config
    :param json_file: the path of the config file
    :return: config object(namespace)
    """
    print("THE Configuration of your experiment ..")

    pprint(vars(config))

    # making sure that you have provided the exp_name.
    try:
        print(" *************************************** ")
        print("The experiment name is {}".format(config.exp_name))
        print(" *************************************** ")
    except AttributeError:
        print("ERROR!!..Please provide the exp_name in json file..")
        exit(-1)

    if config.mode == 'train':
        if config.con_train:
            config.exp_time_load = config.log_time_trained
            log_time = datetime.datetime.now()
            config.exp_time = str(int(time.mktime(log_time.timetuple())))

        elif config.train_TL:
            # pretrain in one setup and fine-tune in another
            config.exp_time_load = config.log_time_trained
            log_time = datetime.datetime.now()
            config.exp_time = str(int(time.mktime(log_time.timetuple())))  # + "/"
            # config.exp_time = log_time.strftime("%H%M-%d%b%Y")
            # config.exp_time = str(int(time.time()))
        else:
            log_time = datetime.datetime.now()
            config.exp_time = str(int(time.mktime(log_time.timetuple()))) #+ "/"
            # config.exp_time = log_time.strftime("%H%M-%d%b%Y")
            # config.exp_time = str(int(time.time()))

    elif config.mode == "test":
        # exp_time = config.log_time
        config.exp_time_load = config.log_time_trained + "/"
        config.exp_time = config.log_time_trained


    if config.con_train:

        config.exp_name_load = os.path.join(f"{config.exp_net}", config.env_name, config.exp_time_load)
        config.checkpoint_dir_load = os.path.join(config.save_data, "experiments", config.exp_name_load, "checkpoints/")
        # config.exp_name = os.path.join(f"{config.exp_net}",  config.env_name, config.exp_time)

        print(" ************** Con train  ********************** ")
        print("The checkpoint_dir_load name is {}".format(config.checkpoint_dir_load))

    
    config.exp_name = os.path.join("{}".format(config.exp_name),  config.env_name, config.exp_time)
    config.checkpoint_dir = os.path.join(config.save_data, "experiments", config.exp_name, "checkpoints/")
    print("The checkpoint_dir name is {}".format(config.checkpoint_dir))

    
    # create some important directories to be used for that experiment.
    config.summary_dir = os.path.join(config.save_data, "experiments", config.exp_name, "summaries/")
    config.out_dir = os.path.join(config.save_data, "experiments", config.exp_name, "out/")
    config.log_dir = os.path.join(config.save_data, "experiments", config.exp_name, "logs/")
    config.save_code_dir = os.path.join(config.save_data, "experiments", config.exp_name, "code/")

    create_dirs([config.summary_dir, config.checkpoint_dir, config.out_dir, config.log_dir, config.save_code_dir])
    # breakpoint()
    Path(config.save_animation).mkdir(parents=True, exist_ok=True)
    Path(config.save_tb_data).mkdir(parents=True, exist_ok=True)
    # model_setup logging in the project
    setup_logging(config.log_dir)

    logging.getLogger().info("Hi, This is root.")
    logging.getLogger().info("After the configurations are successfully processed and dirs are created.")
    logging.getLogger().info("The pipeline of the project will begin now.")

    return config
