import pandas as pd
import numpy as np
import json
import os

Type = ['no_post', 'naive_post', 'mask_prob', 'vocab_embedding']
pathlst = ['nopost/roberta-base_', 'naivefine/ai_unsup_', 'mask/t20058_', 'tokenfine/tk_emb_']
datasets = ['acl_sup', 'sci_sup', 'average']
data = np.zeros((4, 3))

for i, Path in enumerate(pathlst):
    for j, ds in enumerate(datasets[:2]):
        for o in range(1, 6):
            path = os.path.join(Path+ds, str(o), "final_results.json")
            with open(path, 'r') as fr:
                tmp_json = json.load(fr)
                fr.close()

            data[i, j] += tmp_json["predict_macro_f1"]

data /= 5
data[:,2] = (data[:,0]+data[:,1])/2
data = np.around(data*100, 2)

df1 = pd.DataFrame(data,
                   index=Type,
                   columns=datasets)
# df1.to_excel("./test.xlsx") 
print(df1) 
