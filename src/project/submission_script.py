from project.hello_world import hello_world
import sys
import numpy as np
import pandas as pd
from os.path import join
hello_world()

if __name__ == '__main__':
    DATA_PATH = 'data/' if len(sys.argv) == 1 else sys.argv[1]
    test = pd.read_csv(join(DATA_PATH, 'test.csv'))
    dummy_predictions = []
    for test_id in test['id']:
        dummy_predictions.append(f'{np.random.randint(100)} {np.random.randint(101, 200)}')
    test['location'] = dummy_predictions
    submission = test[['id', 'location']]
    submission.to_csv('submission.csv', index=False)