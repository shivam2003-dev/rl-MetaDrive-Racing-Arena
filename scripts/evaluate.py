import sys

from metadrive_racing_arena.cli import main


if __name__ == "__main__":
    main(["evaluate", *sys.argv[1:]])
