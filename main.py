"""
__author__ = "Aneesh Komanduri"

Main
-Capture the config file
-Process the json config passed
-Create an agent instance
-Run the agent
"""

import argparse
from utils.config import *

from agents import *


def main():
    var = False
    # parse the path of the json config file
    arg_parser = argparse.ArgumentParser(description="")
    arg_parser.add_argument(
        'config',
        metavar='config_json_file',
        default='None',
        help='The Configuration file in json format')
    args = arg_parser.parse_args()

    # parse the config json file
    config = process_config(args.config)

    # Create the Agent and pass all the configuration to it then run it..
    agent_class = globals()[config.agent]
    agent = agent_class(config)

    if var:
        agent.load_checkpoint(config.checkpoint_file)
    else:
        agent.run()
        agent.validate()

    agent.finalize()


if __name__ == '__main__':
    main()
