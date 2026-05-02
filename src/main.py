import argparse


def main():
    parser = argparse.ArgumentParser(description="Product title classifier CLI")
    parser.add_argument(
        "action",
        choices=["process", "train", "evaluate", "predict", "serve"],
        help="Action to run: process | train | evaluate | predict | serve",
    )
    args = parser.parse_args()

    if args.action == "process":
        from src.preprocess.process import process

        process()
    elif args.action == "train":
        from src.runner.train import train

        train()
    elif args.action == "evaluate":
        from src.runner.evaluate import run_evaluation

        run_evaluation()
    elif args.action == "predict":
        from src.runner.predict import run_predict

        run_predict()
    elif args.action == "serve":
        from src.web.app import run_app

        run_app()


if __name__ == "__main__":
    main()
