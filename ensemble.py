import pandas as pd
import numpy as np

ensemble_path = 'weights/ensembleb3b5.csv'

paths = [
        'weights/248_efficientnet-b3_f1_test/test_ckpt40.csv',
        'weights/248_efficientnet-b5_f1_test/test_ckpt50.csv'
        ]

labels = np.zeros(15984)
for idx, path in enumerate(paths):
    labels += pd.read_csv(path)['label']

average = np.round(labels / len(paths)).astype('uint8')
df = pd.read_csv(paths[0])
df['label'] = average
df.to_csv(ensemble_path, index=False)
