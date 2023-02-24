"""
The Base Agent class, where all other agents inherit from, that contains definitions for all the necessary functions
"""
import torch
import shutil
import os
import logging
from abc import ABC, abstractmethod


class BaseAgent(ABC):
    """
    This base class will contain the base functions to be overloaded by any agent you will implement.
    """

    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger("Agent")

    def save_checkpoint(self, epoch, is_best=False, lastest=True):
        """
        Checkpoint saver
        :param file_name: name of the checkpoint file
        :param is_best: boolean flag to indicate whether current checkpoint's accuracy is the best so far
        :return:
        """
        if lastest:
            file_name = "checkpoint.pth.tar"
        else:
            file_name = "checkpoint_{:03d}.pth.tar".format(epoch)
        state = {
            'epoch': self.current_epoch + 1,
            'iteration': self.current_iteration,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
        }

        # Save the state
        torch.save(state, os.path.join(self.config.checkpoint_dir, file_name))
        # If it is the best copy it to another file 'model_best.pth.tar'
        if is_best:
            shutil.copyfile(os.path.join(self.config.checkpoint_dir, file_name),
                            os.path.join(self.config.checkpoint_dir, 'model_best.pth.tar'))


    def load_checkpoint(self, checkpoint_dir, epoch, lastest=True, best=False):
        """
        Latest checkpoint loader
        :param file_name: name of the checkpoint file
        :return:
        """
        if lastest:
            file_name = "checkpoint.pth.tar"
        elif best:
            file_name = "model_best.pth.tar"
        else:
            file_name = "checkpoint_{:03d}.pth.tar".format(epoch)
        filename = os.path.join(checkpoint_dir, file_name)
        try:
            self.logger.info("Loading checkpoint '{}'".format(filename))
            # checkpoint = torch.load(filename)
            checkpoint = torch.load(filename, map_location=self.config.device)

            self.current_epoch = checkpoint['epoch']
            self.current_iteration = checkpoint['iteration']
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            if self.scheduler is not None:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            self.logger.info("Checkpoint loaded successfully from '{}' at (epoch {}) at (iteration {})\n"
                             .format(checkpoint_dir, checkpoint['epoch'], checkpoint['iteration']))
        except OSError as e:
            self.logger.info("No checkpoint exists from '{}'. Skipping...".format(self.config.checkpoint_dir))
            self.logger.info("**First time to train**")

    @abstractmethod
    def run(self):
        """
        The main operator
        :return:
        """
        pass

    @abstractmethod
    def finalize(self):
        """
        Finalizes all the operations of the 2 Main classes of the process, the operator and the data loader
        :return:
        """
        pass
