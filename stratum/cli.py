import argparse

def main():
    parser = argparse.ArgumentParser(prog="stratum-eval")
    parser.add_argument("--version", action="version", version="stratum-eval 0.1.0-dev")
    args = parser.parse_args()

if __name__ == "__main__":
    main()
