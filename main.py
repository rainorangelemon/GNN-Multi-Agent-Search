"""
__author__ = "Qingbiao Li and Chenning YU"

Main
-Capture the config file
-Process the json config passed
-Create an agent instance
-Run the agent
"""

import argparse
from utils.config import *
from agents import *

from configs.str2config import str2config, add_default_argument_and_parse

# Pick specific CPU
# os.system("taskset -p -c 0 %d" % (os.getpid()))
# os.system("taskset -p 0xFFFFFFFF %d" % (os.getpid()))
# os.system("taskset -p -c 0-7,16-23 %d" % (os.getpid()))
# os.system("taskset -p -c 8-15,24-31 %d" % (os.getpid()))

# Pick specific GPU
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2, 3"


## Main Pro
def main():
    # parse the path of the json config file
    arg_parser = argparse.ArgumentParser(description="Start the experiment agent")
    config_setup = add_default_argument_and_parse(arg_parser, 'experiment')

    print(config_setup.agent)
    # parse the config json file
    config = process_config(config_setup)

    print(config.seed)

    # set random seed
    seed_everything(config.seed)

    print(config.seed)

    # Create the Agent and pass all the configuration to it then run it..
    agent_class = globals()[config.agent]
    agent = agent_class(config)

    print(agent)

    agent.run()
    agent.finalize()

if __name__ == '__main__':
    main()
