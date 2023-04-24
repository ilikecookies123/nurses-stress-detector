import os
import traceback

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, medfilt


def remove_error(s):
    s['EDA_dx'] = np.abs(s['EDA'].diff())
    s['HR_dx'] = np.abs(s['HR'].diff())
    s['TEMP_dx'] = np.abs(s['TEMP'].diff())
    s['BVP_dx'] = np.abs(s['BVP'].diff())

    EDA_default = s.query('EDA_dx == 0')['EDA'].value_counts().idxmax()
    HR_default = s.query('HR_dx == 0')['HR'].value_counts().idxmax()
    TEMP_default = s.query('TEMP_dx == 0')['TEMP'].value_counts().idxmax()
    try:
        BVP_default = s.query('BVP_dx == 0')['BVP'].value_counts().idxmax()
        r = s.loc[(s['EDA'] != EDA_default) & (s['HR'] != HR_default) & (s['TEMP'] != TEMP_default) & (
                    s['BVP'] != BVP_default)].reset_index(drop=True)
        r = s.reset_index(drop=True)
        return r
    except:
        r = s.loc[(s['EDA'] != EDA_default) & (s['HR'] != HR_default) & (s['TEMP'] != TEMP_default)].reset_index(
            drop=True)
        r = s.reset_index(drop=True)
        return r


def low_pass_filter(signal, cutoff, sampling_frequency, order=4):
    n = 0.5 * sampling_frequency
    norm_cutoff = cutoff / n
    b, a = butter(order, norm_cutoff, btype='low')
    return filtfilt(b, a, signal)


def median_filter(signal, kernal_size):
    return medfilt(signal, kernal_size)


def kalman_filter(signal, initial_state=0.0, initial_covariance=0.1, process_noise=0.01, measurement_noise=0.1):
    state = initial_state
    covariance = initial_covariance

    filtered = np.zeros(len(signal))

    for i, val in enumerate(signal):
        covariance = covariance + process_noise
        kalman_gain = covariance / (covariance + measurement_noise)
        state = state + kalman_gain * (val - state)
        covariance = (1 - kalman_gain) * covariance
        filtered[i] = state

    return filtered


def denoise_dfs(df):
    new_df = {}

    desired = ['X', 'Y', 'Z']
    for x in desired:
        de = df[x].values
        de = low_pass_filter(de, 1.5, 4.0)
        denoised = kalman_filter(de, process_noise=0.0075, measurement_noise=0.3)
        new_df[x] = denoised

    new_df['EDA'] = median_filter(low_pass_filter(df['EDA'].values, 1.5, 4.0), 4001)
    new_df['HR'] = df['HR'].values

    desired = ['TEMP']
    for x in desired:
        de = df[x].values
        de = low_pass_filter(de, 1.5, 4.0)
        denoised = kalman_filter(de, process_noise=0.0075, measurement_noise=0.3)
        new_df[x] = denoised

    new_df['BVP'] = kalman_filter(low_pass_filter(df['BVP'].values, 1.5, 4.0), process_noise=0.0075, measurement_noise=0.3)
    new_df['id'] = df['id']
    new_df['date'] = df['date']
    new_df['time'] = df['time']
    new_df['label'] = df['label']

    return pd.DataFrame(new_df)


def lag_extraction(df):
    lag_HR = -(10 * 4)
    lag_TEMP = -(2 * 4)
    lag_EDA = -(10 * 4)

    df[f'lag_HR'] = df['HR'].shift(lag_HR)
    df[f'lag_TEMP'] = df['TEMP'].shift(lag_TEMP)
    df[f'lag_EDA'] = df['EDA'].shift(lag_EDA)

    return df.dropna(axis=0)


def window_data(df, data_sample_rate=4, window_length=10, overlap_length=0):
    window_size = data_sample_rate * window_length
    overlap = data_sample_rate * overlap_length

    windows = []
    data_length = len(df)
    step = window_size - overlap
    for i in range(0, data_length - window_size + 1, step):
        windows.append(df.iloc[i:i + window_size])
    return windows


def filter_windowed(dfs):
    try:
        stat_features = [col for col in dfs[0].columns if col not in ['id', 'date', 'time', 'label']]
    except:
        return pd.DataFrame()

    pds = []
    for df in dfs:
        di = {}
        for s in stat_features:
            di[s + '_mean'] = df[s].mean()
            di[s + '_std'] = df[s].std()
            di[s + '_min'] = df[s].min()
            di[s + '_max'] = df[s].max()
            di[s + '_dx'] = df.iloc[-1][s] - df.iloc[0][s]

        di['id'] = df.iloc[0]['id']
        di['date'] = df.iloc[-1]['date']
        di['time'] = df.iloc[-1]['time']
        di['label'] = df['label'].value_counts().idxmax()
        pds.append(pd.DataFrame([di]))

    if len(pds) == 0:
        return pd.DataFrame(columns=stat_features)
    return pd.concat(pds)

def run():
    main_df = pd.read_csv('C:\\Users\\hagan\\OneDrive\\Uni Work\\CE888 Data Science and Decision Making\\datasets\\merged_data_labeled.csv')
    split_columns = main_df['datetime'].str.split(' ', expand=True)
    main_df['date'] = split_columns[0]
    main_df['time'] = main_df['datetime']
    main_df.drop('datetime', axis=1, inplace=True)
    main_df = main_df[['X', 'Y', 'Z', 'EDA', 'HR', 'TEMP', 'BVP', 'TI', 'IBI', 'id', 'date', 'time', 'label']]
    main_df['id'] = main_df['id'].astype(str)
    main_df['date'] = pd.to_datetime(main_df['date'])
    main_df['time'] = pd.to_datetime(main_df['time'])


    participants = ['5C', '6B', '6D', '7A', '7E', '8B', '15', '83', '94', 'BG', 'CE', 'DF', 'E4', 'EG', 'F5']
    for p in participants:
        try:
            df = main_df[main_df['id'].isin([p])]
            dfs = {category: df[df['date'] == category] for category in df['date'].unique()}
            new_df = pd.DataFrame()

            for d in df['date'].unique():
                selected_df = dfs[d]
                selected_df['groups'] = (selected_df['time'].diff().dt.seconds > 1).cumsum()

                for g_k, g_df in selected_df.groupby(selected_df['groups']):
                    error_removed = remove_error(g_df)
                    error_removed['groups'] = (error_removed['time'].diff().dt.seconds > 0.1).cumsum()

                    if len(error_removed.groupby(selected_df['groups'])) == 0:
                        denoised = denoise_dfs(error_removed)
                        lag = lag_extraction(denoised)
                        windowed = window_data(lag)
                        extracted = filter_windowed(windowed)
                        if not extracted.empty:
                            new_df = new_df.append(extracted, ignore_index=True)
                    else:
                        for sg_k, er_df in error_removed.groupby(selected_df['groups']):
                            denoised = denoise_dfs(er_df)
                            lag = lag_extraction(denoised)
                            windowed = window_data(lag)
                            extracted = filter_windowed(windowed)
                            if not extracted.empty:
                                new_df = new_df.append(extracted, ignore_index=True)

            directory = "C:/Users/hagan/OneDrive/Uni Work/CE888 Data Science and Decision Making/datasets/cleaned2/"
            file_path = os.path.join(directory, f'{p}_fullpreprocessed.csv')
            print(f'Saving {file_path}')
            new_df.to_csv(file_path, index=False, date_format='%Y-%m-%d %H:%M:%S')
        except:
            print(f"Failed to process {p}")
            traceback.print_exc()

    new_df = pd.DataFrame()
    participants = ['5C', '6B', '6D', '7A', '7E', '8B', '15', '83', '94', 'BG', 'CE', 'DF', 'E4', 'EG', 'F5']
    directory = "C:/Users/hagan/OneDrive/Uni Work/CE888 Data Science and Decision Making/datasets/cleaned/"

    for p in participants:
        file_path = os.path.join(directory, f'{p}_fullpreprocessed.csv')
        d = pd.read_csv(file_path)
        new_df = new_df.append(d, ignore_index=True)

    new_df.to_csv(os.path.join(directory, 'combined_fullpreprocessed.csv'), index=False, date_format='%Y-%m-%d %H:%M:%S')

if __name__ == '__main__':
    run()

