from data_config import *
import os
import json
import pandas as pd

if __name__ == '__main__':
    dic_list = []
    tp,fp,fn=0,0,0
    # print(sorted(os.listdir(exp_dir/'result')))
    for i,dir in enumerate(sorted(map(int,os.listdir(exp_dir/'result')))):
        res = json.load(open(exp_dir/'result' / str(dir) / 'summary.txt'))['bf_result']
        if res['f1']==0.0:
            print(i,dir)
            res = json.load(open(exp_dir/'result' / str(dir) / 'summary.txt'))['epsilon_result']
        tp += res['TP']
        fp += res['FP']
        fn += res['FN']
        res['machine_id'] = i
        dic_list.append(res)
        
    p = tp / (tp+fp)
    r = tp / (tp+fn)
    f1 = 2*p*r / (p+r)
    df = pd.DataFrame(dic_list)
    (exp_dir / 'evaluation_result').mkdir(parents=True, exist_ok=True)
    df.to_csv(exp_dir / 'evaluation_result' / "bf_machine_best_f1.csv", index=False)
    with open(exp_dir / 'evaluation_result' / "bf_res.txt",'a') as fw:
        print(f"all result get_bf f1: {round(f1, 4)} p:{round(p, 4)} r:{round(r, 4)}")
        fw.write(f"all result get_bf f1: {round(f1, 4)} p:{round(p, 4)} r:{round(r, 4)}\n")
        
