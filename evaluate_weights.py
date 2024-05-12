import argparse
import csv
import os
import sys

import numpy as np
import pandas as pd

import utils
import weights

from tqdm import tqdm

BINS = {
    'age': [13, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 81],
    'income': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
    'gender': [0, 1, 5],
    'education': [0, 1, 5],
}
USER_BINS = {
    'age': [0, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 100],
    'income': [0, 10000, 15000, 20000, 35000, 50000, 75000, 100000, 150000, 200000, 1e12],
    'gender': [-100, 0, 100],
    'education': [0, 0.5, 1],
}
USER_BIN_VALUES = {
    # 'age': [0, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 100],
    'income': [0, 10000, 15000, 20000, 35000, 50000, 75000, 100000, 150000, 20000]
    # 'gender': [-100, 0, 100],
    # 'education': [0, 0.5, 1],
}
POPULATION_TABLE_COLS = {
    'age': [
        'total_15to19', 'total_20to24', 'total_25to29', 'total_30to34',
        'total_35to39', 'total_40to44', 'total_45to49', 'total_50to54',
        'total_55to59', 'total_60to64', 'total_65plus',
    ],
    'income': [
        'incomelt10k', 'income10kto14999', 'income15kto24999', 'income25kto34999',
        'income35kto49999', 'income50kto74999', 'income75kto99999',
        'income100kto149999', 'income150kto199999', 'incomegt200k',
    ],
    'gender': ['male_perc', 'female_perc'],
    'education': ['perc_high_school_or_higher', 'perc_bach_or_higher']
}
REDIST_BINS = {
    'age': [0, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 100],
    'income': [0, 30000, 50000, 75000, 1e12],
    'gender': [-100, 0, 100],
    'education': [0, 0.5, 1],
}
REDIST_PERCENTS = {
    'age': [0.4882, 0.3044, 0.1675, 0.4000],
    'income': [0.1989, 0.2047, 0.2428, 0.3536],
    'gender': [0.4878, 0.5122],
    'education': [0.5078, 0.4922],
}


DEMOGRAPHICS = ['income']
# DEMOGRAPHICS = ['age', 'gender']
#DEMOGRAPHICS = ['income', 'education']
# DEMOGRAPHICS = ['age', 'income', 'education']
# DEMOGRAPHICS = ['age', 'gender', 'income', 'education']

USER_TABLE = './data/users_en_30_10pct.csv'
POPULATION_TABLE = './data/acs2015_5yr_age_gender_income_education.csv'
WEIGHTS = './data/weights.csv'

def get_args():
    parser = argparse.ArgumentParser(description='Create Robust post-stratification weights.')
    
    # correction factors
    parser.add_argument('--demographics', type=str, metavar='FIELD(S)', dest='demographics', nargs='+',
        default=DEMOGRAPHICS, help='Field to compare with.')

    # input and output files
    parser.add_argument('--user_table', dest='user_table', type=str, default=USER_TABLE,
        help='User data csv. Default: {d}'.format(d=USER_TABLE))
    parser.add_argument('--population_data', dest='population_data', type=str, default=POPULATION_TABLE,
        help='Population data csv. Default: {d}'.format(d=POPULATION_TABLE))
    parser.add_argument('--weight_table', dest='weight_table', type=str, default=WEIGHTS,
        help='Weights file. Default: {d}'.format(d=WEIGHTS))

    args = parser.parse_args()
    return args


def main(args):
    print('EVALUATING WEIGHTS FOR: {d}'.format(d=args.demographics))
    if len(args.demographics) > 1:
        print("Currently, only single correction factor evaluation is supported.")
        sys.exit(1)

    print('Loading Data')
    
    user_df, population_df = utils.load_data(args.user_table, args.population_data)
    user_df = user_df.reset_index()
    population_df = population_df.reset_index()
    weights_df = pd.read_csv(args.weight_table)
    user_df = user_df.merge(weights_df, on=['user_id','cnty'], how='left')
    # number of users without weights
    print('Number of users without weights:', user_df['weight'].isnull().sum())
    user_df = user_df.dropna(subset=['weight'])
    print('User Data')
    print(user_df)
    print('Population Data')
    print(population_df)
    
    cnty_list = user_df['cnty'].unique().tolist()
    cnty_list.sort()
    size = len(cnty_list)
    print('Number of counties: {d}'.format(d=size))
    
    weighted_diffs = []
    unweighted_diffs = []
    improvements = []

    for cnty in cnty_list:

        try:
            population_data = population_df[population_df['cnty'] == cnty]
        except:
            print('    SKIPPING: population table does not contain county {d}'.format(d=cnty))
            continue
        user_data = user_df[user_df['cnty'] == cnty] 
        
        # print("    User Data")
        # print(user_data)
        # print("    Population Data")
        # print(population_data)
        
        # Currently only one demographic is supported
        demo = args.demographics[0]
        
        # Binned stats
        if demo == 'income':
            population_stats = [population_data[col].values[0] for col in population_data.columns if col.startswith('income')]
            user_stats = [0] * len(population_stats)
            weighted_user_stats = [0] * len(population_stats)
            for _, user in user_data.iterrows():
                user_income = user['income']
                for i, bin in enumerate(USER_BINS['income'][1:]):
                    if user_income <= bin:
                        user_stats[i] += 1
                        weighted_user_stats[i] += user['weight']
                        break
            # renormalize user_stats
            user_stats = [ round(s * 100.0 / np.sum(user_stats),1) for s in user_stats]  
            weighted_user_stats = [ round(s * 100.0 / np.sum(weighted_user_stats),1) for s in weighted_user_stats]      
            
        unweighted_diff = np.mean([abs(a - b) for a, b in zip(user_stats, population_stats)])
        weighted_diff = np.mean([abs(a - b) for a, b in zip(weighted_user_stats, population_stats)])
        improvement = unweighted_diff - weighted_diff
        
        unweighted_diffs.append(unweighted_diff)
        weighted_diffs.append(weighted_diff)
        improvements.append(improvement)
        
        if improvement != 0:
            print("=== County: {} ({} users) ===".format(cnty, len(user_data)))
            print("    Census Bins =        {}".format(population_stats))
            print("    User Bins =          {}".format(user_stats))
            print("    Weighted User Bins = {}".format(weighted_user_stats))
            print("    Diff {} -> Weighted Diff {}".format(unweighted_diff, weighted_diff))
            print("    Improvement: {}".format(improvement))
        
    print('\nDONE Checking Weights')
    
    print('Mean Unweighted Differences', round(np.mean(unweighted_diffs),3))
    print('Mean Weighted Differences', round(np.mean(weighted_diffs),3))
    print('Mean Improvement', round(np.mean(improvements),3) )
    print('Mean Non-Zero Improvement', round(np.mean([i for i in improvements if i != 0]),3) )
    print('Counties that saw no change', np.sum([1 for i in improvements if i == 0]) )
    print('Counties that saw improvement', np.sum([1 for i in improvements if i > 0]) )
    print('Counties that saw degradation', np.sum([1 for i in improvements if i < 0]) )
    
        

if __name__ == '__main__':
    args = get_args()
    main(args)
