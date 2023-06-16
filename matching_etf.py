import os
import copy
import time
import datetime
import glob
import numpy as np
import pandas as pd
import ray
from itertools import combinations
from tqdm import tqdm
from dtaidistance import dtw

from sklearn.preprocessing import MinMaxScaler

KR_ETF_DATA_PATH = '/nas/DnCoPlatformDev/execution/etf_ohlcv/kr'
US_ETF_DATA_PATH = '/nas/DnCoPlatformDev/execution/etf_ohlcv/us'


def dtw_distance(scaled_count, scaled_volume):
    return dtw.distance(scaled_count, scaled_volume)


def compute_metrics(merged_df, kr_isin):
    scaled_kr = MinMaxScaler().fit_transform(
        np.array(merged_df['close_y']).reshape(-1, 1))
    scaled_us = MinMaxScaler().fit_transform(
        np.array(merged_df['close_x']).reshape(-1, 1))
    dtw_dist = dtw_distance(scaled_kr, scaled_us)
    year = (pd.to_datetime(merged_df['Date'].iloc[-1]) -
            pd.to_datetime(merged_df['Date'].iloc[0])).days//365
    ratio = 1 if year == 0 else 1 / year

    first, last = merged_df['close_y'].iloc[0], merged_df['close_y'].iloc[-1]
    cagr_kr = (last/first)**(ratio)-1
    first, last = merged_df['close_x'].iloc[0], merged_df['close_x'].iloc[-1]
    cagr_us = (last/first)**(ratio)-1
    cagr_diff = abs(cagr_kr - cagr_us)

    mdd_kr = (
        (max(merged_df['low_y']) - max(merged_df['high_y']))/max(merged_df['high_y']))*100
    mdd_us = (
        (max(merged_df['low_x']) - max(merged_df['high_x']))/max(merged_df['high_x']))*100
    mdd_diff = abs(mdd_kr - mdd_us)

    total_loss = (dtw_dist + cagr_diff + mdd_diff)/len(merged_df)

    row = [kr_isin, dtw_dist, cagr_kr, cagr_us, mdd_kr,
           mdd_us, cagr_diff, mdd_diff, total_loss]

    return row


@ray.remote
def first_filter(us_etf, kedp):
    kr_etf = pd.read_csv(kedp)
    kr_isin = kedp.split('/')[-1].split('.')[0]
    merged_df = pd.merge(us_etf, kr_etf, how='inner', on='Date')
    if len(merged_df) < 30:
        return None
    metric_result = compute_metrics(merged_df, kr_isin)
    metric_result.append({kr_isin: 1.0})

    return metric_result


@ray.remote
def compute_comb(us_etf, candidate, weight, pool):
    existing_df = pd.DataFrame()
    for isin, existing_weight in pool.items():
        df = pd.read_csv(os.path.join(KR_ETF_DATA_PATH, f'data/{isin}.csv'))
        df['open'] *= existing_weight * (1 - weight)
        df['high'] *= existing_weight * (1 - weight)
        df['low'] *= existing_weight * (1 - weight)
        df['close'] *= existing_weight * (1 - weight)
        existing_df = pd.concat([existing_df, df])
    cand_df = pd.read_csv(os.path.join(
        KR_ETF_DATA_PATH, f'data/{candidate}.csv'))
    cand_df['open'] *= weight
    cand_df['high'] *= weight
    cand_df['low'] *= weight
    cand_df['close'] *= weight
    kr_df = pd.concat([existing_df, cand_df])
    kr_df = kr_df.groupby('Date').sum().reset_index()

    merged_df = pd.merge(kr_df, us_etf, on='Date')

    result = compute_metrics(merged_df, candidate)
    result.append(weight)

    return result


def filter_top_k(results, metrics, k):
    # filter considering sector, etc...

    for metric in metrics:
        results = results.sort_values(by=[metric])
        results[f'{metric}_ranking'] = range(len(results))

    metric_list = [f'{metric}_ranking' for metric in metrics]
    results['total_ranking'] = results[metric_list].sum(axis=1)
    results = results.sort_values(by=['total_ranking'])

    top_k_results = results.iloc[:k]

    return top_k_results


def add_isin_to_pool(pool, new_isin, weight):
    for existing_isin in pool.keys():
        pool[existing_isin] *= 1 - weight
    pool[new_isin] = weight

    return pool


def get_closest_kr_etf(us_etf, kr_etf_candidates, cols, pool):
    results = pd.DataFrame(columns=cols)

    done_checker_ids = []
    for candidate in kr_etf_candidates:
        for cand_weight in range(1, 10):
            done_checker_id = compute_comb.remote(
                us_etf, candidate, cand_weight * 0.1, pool)
            done_checker_ids.append(done_checker_id)

    total_len = len(done_checker_ids)
    while len(done_checker_ids):
        print(
            f'[get_closest_kr_etf] {round(1-len(done_checker_ids)/total_len, 2)} % / processing: {total_len-len(done_checker_ids)} / {total_len}')
        done_id, done_checker_ids = ray.wait(done_checker_ids)
        row = ray.get(done_id[0])
        if row is not None:
            results = pd.concat([results, pd.DataFrame(
                [row], columns=cols)], ignore_index=True)

    # metrics = ['dtw', 'cagr', 'mdd']
    # closest = filter_top_k(results, metrics, k=1)
    closest = results.sort_values(by=['total_loss']).iloc[:1]

    return closest


def main(us_ticker_list, matching_dict):
    ray.init()
    start_dt = datetime.datetime.now()

    for us_ticker in tqdm(us_ticker_list):

        # ray.init(local_mode=True)
        us_etf = pd.read_csv(os.path.join(
            US_ETF_DATA_PATH, f'data/{us_ticker}.csv'))
        cols = ['kr_etf', 'dtw', 'cagr_kr', 'cagr_us', 'mdd_kr',
                'mdd_us', 'cagr', 'mdd', 'total_loss', 'weight']
        kedp_list = glob.glob(os.path.join(KR_ETF_DATA_PATH, 'data/*.csv'))

        #------------------------- first filtering -------------------------#
        done_checker_ids = [first_filter.remote(
            us_etf, kedp) for kedp in kedp_list]
        results = pd.DataFrame([], columns=cols)

        total_len = len(kedp_list)
        while len(done_checker_ids):
            print(
                f'[first_filter] elapsed time: {datetime.datetime.now() - start_dt} / processing: {total_len-len(done_checker_ids)} / {total_len}')
            done_id, done_checker_ids = ray.wait(done_checker_ids)
            row = ray.get(done_id[0])
            if row is not None:
                results = pd.concat([results, pd.DataFrame(
                    [row], columns=cols)], ignore_index=True)

        #------------------------- filter top k -------------------------#
        metrics = ['dtw', 'cagr', 'mdd']
        K = 10
        top_k_results = filter_top_k(results, metrics, K)
        top_k_results = top_k_results.reset_index(drop=True)

        if len(top_k_results) <= 0:
            continue

        main_ticker = matching_dict.get(us_ticker, None)
        print(main_ticker)
        #------------------------- add feasible sub items -------------------------#
        final_df = pd.DataFrame([], columns=cols)
        if main_ticker is not None:
            main_ticker_row = results[results['kr_etf'] == main_ticker]
            if main_ticker in top_k_results['kr_etf'].tolist():
                cur_metric = main_ticker_row.iloc[0][metrics]
            else:
                top_k_results = pd.concat(
                    [main_ticker_row, top_k_results], ignore_index=True)
                cur_metric = top_k_results.iloc[0][metrics]
        else:
            cur_metric = top_k_results.iloc[0][metrics]

        add_count = 0
        it = 1
        while True:
            print(
                    f'[kr_etf_candidates] elapsed time: {datetime.datetime.now() - start_dt} {it} iter')
            for _, tk_row in top_k_results.iterrows():
                pool = tk_row['weight']
                
                if main_ticker and main_ticker not in tk_row['weight'].keys():
                    pass
                else:
                    final_df = pd.concat(
                        [final_df, pd.DataFrame([tk_row], columns=cols)])
                kr_etf_candidates = [
                    i for i in top_k_results['kr_etf'] if i not in pool.keys()]
                if len(kr_etf_candidates) <= 0:
                    break
                closest = get_closest_kr_etf(
                    us_etf, kr_etf_candidates, cols, pool)
                for _, closest_row in closest.iterrows():
                    next_ = False
                    for metric in metrics:
                        if closest_row[metric] <= cur_metric[metric]:
                            next_ = True
                            break
                    if next_:
                        tmp_pool = copy.deepcopy(pool)
                        tmp_pool = add_isin_to_pool(
                            tmp_pool, closest_row['kr_etf'], closest_row['weight'])
                        if main_ticker and main_ticker not in tmp_pool.keys():
                            continue
                        closest_row['weight'] = tmp_pool
                        final_df = pd.concat([final_df,
                                              pd.DataFrame([closest_row], columns=cols)], ignore_index=True)
            if (len(final_df) - add_count) > 10:
                add_count = len(final_df)
                final_df = final_df.sort_values('total_loss')
            else:
                break
            if add_count:
                K = 10
                top_k_results, _ = filter_top_k(final_df, metrics, K)
                cur_metric = top_k_results.iloc[0][metrics]
            it+=1

        # final_df.to_csv(f'/nas/DnCoPlatformDev/execution/etf_ohlcv/result/{us_ticker}_results.csv')
        final_df = final_df.sort_values('total_loss')
        final_df.to_csv(f'./results/{us_ticker}_results.csv')

    ray.shutdown()


if __name__ == '__main__':
    # us_ticker_list = ['VWO']
    # matching 되어 있는 정보를 불러와서 main_ticker로 지정해야 할듯
    # 매번 입력받는 방식 x
    #matching_ticker = pd.read_csv('./fint_matching.csv', index_col=0)

    us_ticker_list = []
    matching_dict = dict()
    for us_ticker_path in glob.glob(US_ETF_DATA_PATH + '/data/*.csv'):
        us_ticker = os.path.basename(us_ticker_path).strip('.csv')
        us_ticker_list.append(us_ticker)
        # if us_ticker in matching_ticker['us_etf']:
        #     matching_dict[us_ticker] = matching_ticker[matching_ticker['us_etf']==us_ticker]['kr_etf']
    # matching_dict['SPDW']='195970'
    # matching_dict['SPAB']='273130'
    # matching_dict['SPTS']='305080'
    # us_ticker_list = ['SPTS']
    matching_dict['RWO']='182480'
    us_ticker_list = ['RWO']
    main(us_ticker_list, matching_dict)
