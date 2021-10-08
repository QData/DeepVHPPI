
import lmdb 
import pickle as pkl 
import json
import numpy as np

for split in ['train','valid','test']:


    env = lmdb.open('contact/proteinnet_'+split+'.lmdb') 

    output_list = []
    with env.begin() as txn: 
        num_examples = pkl.loads(txn.get(b'num_examples'))
        for idx in range(int(num_examples)):
            sample = pkl.loads(txn.get(str(idx).encode()))
            sample['id'] = idx
            for key,val in sample.items():
                if isinstance(val, np.ndarray):
                    sample[key] = val.tolist()
            output_list.append(sample)
 
    with open('contact/proteinnet_'+split+'.json', 'w') as fout:
        json.dump(output_list , fout)

# ter = np.array(lines[0]['tertiary']) 
# contact_map = np.less(squareform(pdist(ter)), 8.0).astype(np.int64) 