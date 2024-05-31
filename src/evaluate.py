import torch

def evalRecall(data_list):
    mentions_2_result={}
    for item in data_list:
        if item['origin_mention'] not in mentions_2_result.keys():
            mentions_2_result[item['origin_mention']] = {
                "true":item['origin_normalized_result'],
                "pred":[]
            }
        mentions_2_result[item['origin_mention']]["pred"].extend(item['candidates'])
    total,hit = 0,0
    for key,value in mentions_2_result.items():
        pred = list(set(value['pred']))
        true = value['true']
        total += 1
        if set(true).issubset(set(pred)):
            hit += 1
        # print(pred,true,hit)
    return hit/total

def evalacc(predss, truess):
    total, hit, mul_total, mul_hit, uni_total, uni_hit = 0, 0, 0, 0, 0, 0
    for preds, trues in zip(predss, truess):
        # print(preds, trues)
        total += 1
        if len(trues) > 1:
            mul_total += 1
            if set(trues) == set(preds):
                mul_hit += 1
                hit += 1
        else:
            uni_total += 1
            if set(trues) == set(preds):
                uni_hit += 1
                hit += 1
    return hit/total, mul_hit/mul_total, uni_hit/uni_total


def evalf1(predss, truess):
    k, m, n = 0, 0, 0
    for preds, trues in zip(predss, truess):
        # print(preds, trues)
        m += len(trues)
        n += len(preds)
        for pred in preds:
            if pred in trues:
                k += 1
    p = k/(n+1e-19)
    r = k/(m+1e-19)
    f1 = 2*p*r / (p+r+1e-19)
    return f1, p, r

def evalNumAcc(predss, truess):
    total, hit = 0, 0
    for preds, trues in zip(predss, truess):
        # print(preds, trues)
        total += 1
        if len(trues) == len(preds):
            hit += 1
    return hit/total

def compute_metrics(eval_pred):
    pred_index,labels = eval_pred[0]
    result = 0.
    for pd, gd in zip(pred_index,labels):
        if set([i for i in gd if i!=-1]).issubset(set(pd[:10])):
            result += 1

    return {
        "hit_rate": float(
            result / len(golden_index)
        )
    }