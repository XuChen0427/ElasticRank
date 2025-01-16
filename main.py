import numpy as np
import argparse
import yaml
from reranker import RecReRanker
#from
import os


if __name__ == "__main__":
# Initialize ArgumentParser
    parser = argparse.ArgumentParser(description="ElasticRank.")

    # add parameters
    parser.add_argument("--dataset", type=str, choices=["steam"], default="steam", help="your dataset")
    parser.add_argument("--train_config_file", type=str, default="train_Reranking.yaml", help="your train yaml file")
    #parser.add_argument("--reprocess", type=str, choices=["yes", "no"], default="no", help="your dataset")
    #parser.add_argument("topk", type=float, default=10, help="ranking size")
    args = parser.parse_args()
    with open(os.path.join(args.train_config_file), 'r') as f:
        train_config = yaml.safe_load(f)
    train_config['dataset'] = args.dataset
    print("your training config...")
    print(train_config)
    # parse the args

    print("your args:", args)

    reranker = RecReRanker(args.dataset, train_config)
    reranker.rerank()



