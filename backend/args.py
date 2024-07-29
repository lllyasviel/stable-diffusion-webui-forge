import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--cuda-stream", action="store_true")

args = parser.parse_known_args()[0]
