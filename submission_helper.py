import pandas as pd
import numpy as np
def create_submission(pred_sub, name_of_the_file='submission'):
    """
    Writes the submission in a csv file

    INPUT:
        pred_sub              - The list of predictions
        name_of_the_file      - (optional): the path of the file

    """

    df_sub = pd.DataFrame(pred_sub, columns=['Prediction'])
    df_sub.index.name = 'Id'
    df_sub.index = np.arange(1, 10001)
    df_sub[df_sub['Prediction'] == 0] = -1
    df_sub.to_csv(name_of_the_file + '.csv',index_label='Id')

    print('submission file created as "'+ name_of_the_file+'.csv"')
