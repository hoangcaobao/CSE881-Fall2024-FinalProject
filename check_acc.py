import pandas as pd
if __name__ == "__main__":
    results = pd.read_csv("result_llava_next.csv")
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
