"""
This Python module contains a collection of utility functions that enable various 
operations across different tasks and applications. It offers a range of functions to 
simplify common tasks and streamline complex operations in a modular way.
"""

import logging
import yaml

#------------------------------------------------------------------------------------

def green_message(message: str) -> str:
    return "\x1b[32m" + message + "\x1b[0m"

#------------------------------------------------------------------------------------

def init_logging() -> None:
    logging.basicConfig(
        level=logging.INFO, 
        format="%(asctime)s - %(levelname)-8s - %(message)s", 
        datefmt="%Y-%m-%d %H:%M:%S"
    )

#------------------------------------------------------------------------------------

def read_config(path: str) -> dict:
    with open(path, "r") as file:
        configs = yaml.load(file, Loader=yaml.FullLoader)
    return configs