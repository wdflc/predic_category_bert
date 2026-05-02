from runner.train import train
from runner.predict import run_predict
from runner.evaluate import run_evaluation
from web.app import run_app
import argparse

def main():
 parser = argparse.ArgumentParser(description="商品标题分类器 CLI")
 parser.add_argument(
  "action",
  choices=["process", "train", "evaluate", "predict", "serve"],
  help="操作类型：process | train | evaluate | predict | serve"
 )
 args = parser.parse_args()

 if args.action == "process":
  from preprocess.process import process
  process()

 elif args.action == "train":
  from runner.train import train
  train()

 elif args.action == "evaluate":
  from runner.evaluate import run_evaluation
  run_evaluation()

 elif args.action == "predict":
  from runner.predict import run_predict
  run_predict()

 elif args.action == "serve":
  from web.app import run_app
  run_app()

 else:
  print("未知操作类型，请选择：process / train / evaluate / predict / serve")

if __name__ == "__main__":
 main()

if __name__ == '__main__':
    train()
    # run_predict()
    # run_evaluation()
    # run_app()
