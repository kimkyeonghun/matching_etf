import glob
import pandas as pd
import datetime
import numpy as np

from sklearn.preprocessing import MinMaxScaler

start_date = datetime.date.today() -datetime.timedelta(days=14)
end_date = datetime.date.today()


def weight_function(x):
    return np.exp(-0.1*x)

def news_scoring():
    score_df = pd.DataFrame([], columns=['ticker', 'score'])

    for file_name in glob.glob('./news_dataset/en/yahoo_feed/*.csv'):
        df = pd.read_csv(file_name, index_col=0)

        df['published'] = pd.to_datetime(df['published'])
        df['published_date'] = df['published'].dt.date
        df: pd.DataFrame = df[(df['published_date'] < end_date) & (df['published_date'] >= start_date)]
        if len(df)==0:
            continue
        ticker: str = df.topic.iloc[0]
        ticker = ticker.replace('$', '')

        tmp_df = df.set_index('published')
        
        df_test_sample = tmp_df.resample('1D').count()  # 일 단위 빈도수 리샘플링
        counting_ticker = df_test_sample['topic'][::-1]

        score = 0
        raw_score = 0
        for i in range(len(counting_ticker)):
            diff_day = (end_date - datetime.date.fromtimestamp(pd.Timestamp(counting_ticker.index[i]).timestamp())).days
            #diff_day = (end_date - pd.Timestamp(counting_ticker.index[i])).days
            score += weight_function(diff_day-1)*counting_ticker.iloc[i]
            raw_score += counting_ticker.iloc[i]

        score_df = pd.concat([
            score_df,
            pd.DataFrame({
            'ticker': [ticker],
            'score' : [score],
            'raw_counting': [raw_score]
            })
        ], ignore_index=True)
    scaler = MinMaxScaler(feature_range=(0, 10))
    results = scaler.fit_transform(np.array(np.log(score_df['score'].tolist())).reshape(-1,1))
    score_df['log_score'] = list(map(lambda x: np.round(x, 3)[0], results))
    sorted_score_df = score_df.sort_values('log_score')
    
    return sorted_score_df


def tweet_scoring():

    file_list = glob.glob('./stocktwits/*.csv')

    total = pd.DataFrame([], columns=['id','body','created_at','user.username','entities.sentiment.basic','ticker'])
    for file_name in file_list:
        df = pd.read_csv(file_name, index_col=0)
        total = pd.concat([total, df], ignore_index=True)
    
    score_df = pd.DataFrame([], columns=['ticker','score'])
    for f_l in file_list:
        df_test = pd.read_csv(f_l, index_col=0)
        df_test['created_at'] = pd.to_datetime(df_test['created_at'])
        df_test['created_at_date'] = df_test['created_at'].dt.date
        df_test = df_test[(df_test['created_at_date'] < end_date) & (df_test['created_at_date'] >= start_date)]
        if len(df_test)==0:
            continue
        ticker = df_test['ticker'].iloc[0]
        if ticker==True:
            ticker='TRUE'
            print(ticker)
        tmp_df = df_test.set_index('created_at')

        df_test_sample = tmp_df.resample('1D').count()  # 일 단위 빈도수 리샘플링
        counting_ticker = df_test_sample['ticker'][::-1]

        score = 0
        raw_score = 0
        for i in range(len(counting_ticker)):
            diff_day = (end_date - datetime.date.fromtimestamp(pd.Timestamp(counting_ticker.index[i]).timestamp())).days
            #diff_day = (end_date - pd.Timestamp(counting_ticker.index[i])).days
            score += weight_function(diff_day-1)*counting_ticker.iloc[i]
            raw_score += counting_ticker.iloc[i]
        score_df = pd.concat([
            score_df,
            pd.DataFrame({
            'ticker': [ticker],
            'score' : [score],
            'raw_counting': [raw_score]
            })
        ], ignore_index=True)

    scaler = MinMaxScaler(feature_range=(0, 10))
    results = scaler.fit_transform(np.array(np.log(score_df['score'].tolist())).reshape(-1,1))
    score_df['log_score'] = list(map(lambda x: np.round(x, 3)[0], results))
    sorted_score_df = score_df.sort_values('log_score')

    return sorted_score_df


def merge_score(news_score, tweet_score):
    df = pd.read_excel('v2_all_2.xlsx', index_col=0)

    q3 = int(len(df)*0.7)
    df = df[:q3]

    ticker_list = df['ticker'].tolist()

    news_total_score = []
    tweet_total_score = []
    for ticker in ticker_list:
        try:
            news_total_score.append(news_score[news_score.ticker==ticker]['log_score'].values[0])
        except:
            news_total_score.append(0)
        try:
            tweet_total_score.append(tweet_score[tweet_score.ticker==ticker]['log_score'].values[0])
        except:
            tweet_total_score.append(0)

    total_score_df = pd.DataFrame([])
    total_score_df['ticker'] = ticker_list
    total_score_df['news'] = news_total_score
    total_score_df['tweet'] = tweet_total_score
    total_score_df['mean_score'] = (total_score_df['news'] + total_score_df['tweet'])/2

    return total_score_df
    

if __name__ == '__main__':
    print(start_date, end_date)
    news_score = news_scoring()
    tweet_score = tweet_scoring()
    total_score = merge_score(news_score, tweet_score)
    total_score.to_csv('concern_score.csv')
    print(total_score[total_score['mean_score']!=0].sort_values('mean_score', ascending=False))
    print(total_score[total_score['mean_score']!=0].shape)