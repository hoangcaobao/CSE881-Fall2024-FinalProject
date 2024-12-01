import pandas as pd
import argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--result_file', default="result_llava_next.csv", type=str, required=True)
    args = parser.parse_args()

    results = pd.read_csv(args.result_file)
    total = len(results)
    acc = 0
    for i, result in results.iterrows():
        label = result["label"]
        pred = result["predict"]
        
        format_answer = "Answer:"
        find_format = pred.find(format_answer)
        if find_format != -1:
            pred = pred[find_format + len(format_answer): ]
            
        pred = "".join(pred.split(" "))
        acc += 1 if label.lower() == pred.lower() else 0
    print(acc / total)
