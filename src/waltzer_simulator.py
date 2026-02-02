import sys
from loaders.input_loader import *


def main():
    try:
        params = load_parameters("parameters.txt")
    except ValueError as e:
        print(f"Input error: {e}")
        sys.exit(1)

    print(params)
    
if __name__ == "__main__":
    main()
