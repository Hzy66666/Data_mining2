import pandas as pd
from progressbar import *
import os
import json
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


min_sup = 0.1
min_conf = 0.5


Property_list = ['location', 'Area Id', 'beat', 'Priority', 'Incident Type Id', 'Event Number']
result_path = './results'


def apriori(dataset):             
    C1 = C1_generation(dataset)      
    dataset = [set(data) for data in dataset]
    F1, sup_rata = Ck_low_support_filtering(dataset, C1)
    F = [F1]
    k = 2
    while len(F[k-2]) > 0:
        Ck = apriori_gen(F[k-2], k)        
        Fk, support_k = Ck_low_support_filtering(dataset, Ck)    
        sup_rata.update(support_k)
        F.append(Fk)
        k += 1
    return F, sup_rata

def C1_generation(dataset):       
    C1 = []
    progress = ProgressBar()
    for data in progress(dataset):
        for item in data:
            if [item] not in C1:
                C1.append([item])
    return [frozenset(item) for item in C1]

def Ck_low_support_filtering(dataset, Ck):       
    Ck_count = dict()
    for data in dataset:
        for cand in Ck:
            if cand.issubset(data):
                if cand not in Ck_count:
                    Ck_count[cand] = 1
                else:
                    Ck_count[cand] += 1

    num_items = float(len(dataset))
    return_list = []
    sup_rata = dict()
    for key in Ck_count:
        support  = Ck_count[key] / num_items
        if support >= min_sup:
            return_list.insert(0, key)
        sup_rata[key] = support
    return return_list, sup_rata

def apriori_gen(Fk, k):       
    return_list = []
    len_Fk = len(Fk)

    for i in range(len_Fk):
        for j in range(i+1, len_Fk):
            F1 = list(Fk[i])[:k-2]
            F2 = list(Fk[j])[:k-2]
            F1.sort()
            F2.sort()
            if F1 == F2:
                return_list.append(Fk[i] | Fk[j])
    return return_list

def generate_rules(F, sup_rata):

    strong_rules_list = []
    for i in range(1, len(F)):
        for freq_set in F[i]:
            H1 = [frozenset([item]) for item in freq_set]
            if i > 1:
                rules_from_reasoned_item(freq_set, H1, sup_rata, strong_rules_list)
            else:
                cal_conf(freq_set, H1, sup_rata, strong_rules_list)
    return strong_rules_list

def rules_from_reasoned_item(freq_set, H, sup_rata, strong_rules_list):
    m = len(H[0])
    if len(freq_set) > (m+1):
        Hmp1 = apriori_gen(H, m+1)
        Hmp1 = cal_conf(freq_set, Hmp1, sup_rata, strong_rules_list)
        if len(Hmp1) > 1:
            rules_from_reasoned_item(freq_set, Hmp1, sup_rata, strong_rules_list)

def cal_conf(freq_set, H, sup_rata, strong_rules_list):  
    prunedH = []
    for reasoned_item in H:
        sup = sup_rata[freq_set]
        conf = sup / sup_rata[freq_set - reasoned_item]
        lift = conf / sup_rata[reasoned_item]
        jaccard = sup / (sup_rata[freq_set - reasoned_item] + sup_rata[reasoned_item] - sup)
        if conf >= min_conf:
            strong_rules_list.append((freq_set-reasoned_item, reasoned_item, sup, conf, lift, jaccard))
            prunedH.append(reasoned_item)
    return prunedH

def data_read():

    data2011 = pd.read_csv("data/records-for-2011.csv", encoding="utf-8")
    data2012 = pd.read_csv("data/records-for-2012.csv", encoding="utf-8")
    data2013 = pd.read_csv("data/records-for-2013.csv", encoding="utf-8")
    data2014 = pd.read_csv("data/records-for-2014.csv", encoding="utf-8")
    data2015 = pd.read_csv("data/records-for-2015.csv", encoding="utf-8")
    data2016 = pd.read_csv("data/records-for-2016.csv", encoding="utf-8")

    data2012.rename(columns={"Location 1": "Location"}, inplace = True)
    data2013.rename(columns={"Location ": "Location"}, inplace = True)
    data2014.rename(columns={"Location 1": "Location"}, inplace = True)

    data2011_temp = data2011[["Agency", "Location", "Area Id", "Beat", "Priority", "Incident Type Id", "Incident Type Description", "Event Number"]]
    data2012_temp = data2012[["Agency", "Location", "Area Id", "Beat", "Priority", "Incident Type Id", "Incident Type Description", "Event Number"]]
    data2013_temp = data2013[["Agency", "Location", "Area Id", "Beat", "Priority", "Incident Type Id", "Incident Type Description", "Event Number"]]
    data2014_temp = data2014[["Agency", "Location", "Area Id", "Beat", "Priority", "Incident Type Id", "Incident Type Description", "Event Number"]]
    data2015_temp = data2015[["Agency", "Location", "Area Id", "Beat", "Priority", "Incident Type Id", "Incident Type Description", "Event Number"]]
    data2016_temp = data2016[["Agency", "Location", "Area Id", "Beat", "Priority", "Incident Type Id", "Incident Type Description", "Event Number"]]

    data_all = pd.concat([data2011_temp, data2012_temp, data2013_temp, data2014_temp, data2015_temp, data2016_temp],
                            axis=0)
    data_all = data_all.dropna(how='any')

    print(len(data_all))
    return data_all.head(100000)


def mining(feature_list):
        out_path = result_path

        data_all = data_read()
        rows = data_all.values.tolist()

        # 将数据转为数据字典存储
        dataset = []
        feature_names = ["Agency", "Location", "Area Id", "Beat", "Priority", "Incident Type Id", "Incident Type Description", "Event Number"]
        for data_line in rows:
            data_set = []
            for i, value in enumerate(data_line):
                if not value:
                    data_set.append((feature_names[i], 'NA'))
                else:
                    data_set.append((feature_names[i], value))
            dataset.append(data_set)

        # 获取频繁项集
        freq_set, sup_rata = apriori(dataset)
        sup_rata_out = sorted(sup_rata.items(), key=lambda d: d[1], reverse=True)
        print("sup_rata ", sup_rata)
        # 获取强关联规则列表
        strong_rules_list = generate_rules(freq_set, sup_rata)
        strong_rules_list = sorted(strong_rules_list, key=lambda x: x[3], reverse=True)
        print("strong_rules_list ", strong_rules_list)

        # 将频繁项集输出到结果文件
        freq_set_file = open(os.path.join(out_path, 'freq_lists.json'), 'w')
        for (key, value) in sup_rata_out:
            result_dict = {'set': None, 'sup': None}
            set_result = list(key)
            sup_result = value
            if sup_result < min_sup:
                continue
            result_dict['set'] = set_result
            result_dict['sup'] = sup_result
            json_str = json.dumps(result_dict, ensure_ascii=False)
            freq_set_file.write(json_str + '\n')
        freq_set_file.close()

        # 将关联规则输出到结果文件
        rules_file = open(os.path.join(out_path, 'rules.json'), 'w')
        for result in strong_rules_list:
            result_dict = {'X_set': None, 'Y_set': None, 'sup': None, 'conf': None, 'lift': None, 'jaccard': None}
            X_set, Y_set, sup, conf, lift, jaccard = result
            result_dict['X_set'] = list(X_set)
            result_dict['Y_set'] = list(Y_set)
            result_dict['sup'] = sup
            result_dict['conf'] = conf
            result_dict['lift'] = lift
            result_dict['jaccard'] = jaccard

            json_str = json.dumps(result_dict, ensure_ascii=False)
            rules_file.write(json_str + '\n')
        rules_file.close()

def Visualization():

    with open("./results/freq_lists.json") as f1:
        freq = [json.loads(each) for each in f1.readlines()]

    with open("./results/rules.json") as f2:
        rules = [json.loads(each) for each in f2.readlines()]

    freq_sup = [each["sup"] for each in freq]
    plt.boxplot(freq_sup)
    plt.ylabel("Frequent item")
    plt.show()

    rules_sup = [each["sup"] for each in rules]
    rules_conf = [each["conf"] for each in rules]


    plt.scatter(rules_sup, rules_conf, marker='o', color='red', s=40)
    plt.xlabel = 'Sup'
    plt.ylabel = 'Conf'
    plt.legend(loc='best')
    plt.show()


if __name__ == "__main__":
    # mining(Property_list)
    Visualization()

