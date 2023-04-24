import os
import shutil
import multiprocessing
import numpy as np
import pandas as pd
from datetime import timedelta, datetime

"""
From Assignment 1
Based off code handed out for Assignment 1
"""


class Loader:
    CONTENT_ROOT = ''
    STRESS_DATA_PATH = 'Data/Stress_dataset'
    COMBINED_DATA_SAVE_LOCAL = 'processed_data'
    MERGED_DATA_SAVE_LOCAL = 'merged_data'
    SURVEY_PATH = 'Data/SurveyResults.xlsx'

    DESIRED_SIGNALS = ['ACC.csv', 'EDA.csv', 'HR.csv', 'TEMP.csv', 'BVP.csv', 'IBI.csv']
    SIGNALS = ['acc', 'eda', 'hr', 'temp', 'bvp', 'ibi']
    COLUMNS = ['X', 'Y', 'Z', 'EDA', 'HR', 'TEMP', 'BVP', 'TI', 'IBI', 'id', 'datetime']

    def reset_dataset(self):
        print("unzipping...")
        self.unzip()
        print("combining...")
        self.combine_data()
        print("merging...")
        self.merge_data()
        print("labelling...")
        self.label_data()
        print("Done!")

    def unzip_parallel(self, file, sub_file):
        shutil.unpack_archive(
            os.path.join(self.STRESS_DATA_PATH, file, sub_file),
            os.path.join(self.STRESS_DATA_PATH, file, sub_file[:-4])
        )

    def process_df(self, df, file):
        start_timestamp = df.iloc[0, 0]
        sample_rate = df.iloc[1, 0]
        new_df = pd.DataFrame(df.iloc[2:].values, columns=df.columns)
        new_df['id'] = file[-2:]
        new_df['datetime'] = [(start_timestamp + i / sample_rate) for i in range(len(new_df))]
        return new_df

    def merge_parallel(self, id, acc, eda, hr, temp, bvp, ibi):
        print(f"Processing {id}")
        columns = self.COLUMNS

        df = pd.DataFrame(columns=columns)

        acc_id = acc[acc['id'] == id]
        eda_id = eda[eda['id'] == id].drop(['id'], axis=1)
        hr_id = hr[hr['id'] == id].drop(['id'], axis=1)
        temp_id = temp[temp['id'] == id].drop(['id'], axis=1)
        bvp_id = bvp[bvp['id'] == id].drop(['id'], axis=1)
        ibi_id = ibi[ibi['id'] == id].drop(['id'], axis=1)

        df = acc_id.merge(eda_id, on='datetime', how='outer')
        df = df.merge(temp_id, on='datetime', how='outer')
        df = df.merge(hr_id, on='datetime', how='outer')
        df = df.merge(bvp_id, on='datetime', how='outer')
        df = df.merge(ibi_id, on='datetime', how='outer')

        df.fillna(method='ffill', inplace=True)
        df.fillna(method='bfill', inplace=True)

        return df

    def parallel(self, id, df, survey_df):
        columns = self.COLUMNS + ['label']
        new_df = pd.DataFrame(columns=columns)

        sdf = df[df['id'] == id].copy()
        survey_sdf = survey_df[survey_df['ID'] == id].copy()

        for _, survey_row in survey_sdf.iterrows():
            ssdf = sdf[
                (sdf['datetime'] >= survey_row['Start datetime']) & (
                        sdf['datetime'] <= survey_row['End datetime'])].copy()

            if not ssdf.empty:
                ssdf['label'] = np.repeat(survey_row['Stress level'], len(ssdf.index))
                new_df = pd.concat([new_df, ssdf], ignore_index=True)
            else:
                print(
                    f"{survey_row['ID']} is missing label {survey_row['Stress level']} at {survey_row['Start datetime']} to {survey_row['End datetime']}")

        return new_df

    def unzip(self):
        shutil.unpack_archive(self.CONTENT_ROOT + 'Data.zip', self.CONTENT_ROOT + 'Data')
        shutil.unpack_archive(self.CONTENT_ROOT + 'Data/Stress_dataset.zip', self.CONTENT_ROOT + 'Data/Stress_dataset')

        stress_data_path = self.CONTENT_ROOT + self.STRESS_DATA_PATH

        cpu_count = int(multiprocessing.cpu_count() / 2)

        new_list = [
            (file, sub_file)
            for file in os.listdir(stress_data_path)
            for sub_file in os.listdir(os.path.join(stress_data_path, file))
        ]

        pool = multiprocessing.Pool(cpu_count)
        results = pool.starmap(self.unzip_parallel, new_list)
        pool.close()

    def combine_data(self):

        try:
            os.mkdir(self.COMBINED_DATA_SAVE_LOCAL)
        except:
            pass

        final_columns = {
            'ACC': ['id', 'X', 'Y', 'Z', 'datetime'],
            'EDA': ['id', 'EDA', 'datetime'],
            'HR': ['id', 'HR', 'datetime'],
            'TEMP': ['id', 'TEMP', 'datetime'],
            'BVP': ['id', 'BVP', 'datetime'],
            'IBI': ['id', 'TI', 'IBI', 'datetime']
        }

        names = {
            'ACC.csv': ['X', 'Y', 'Z'],
            'EDA.csv': ['EDA'],
            'HR.csv': ['HR'],
            'TEMP.csv': ['TEMP'],
            'BVP.csv': ['BVP'],
            'IBI.csv': ['TI', 'IBI']
        }

        desired_signals = self.DESIRED_SIGNALS

        acc = pd.DataFrame(columns=final_columns['ACC'])
        eda = pd.DataFrame(columns=final_columns['EDA'])
        hr = pd.DataFrame(columns=final_columns['HR'])
        temp = pd.DataFrame(columns=final_columns['TEMP'])
        bvp = pd.DataFrame(columns=final_columns['BVP'])
        ibi = pd.DataFrame(columns=final_columns['IBI'])

        DATA_PATH = self.CONTENT_ROOT + self.STRESS_DATA_PATH

        for file in os.listdir(DATA_PATH):
            print(f'Processing {file}')
            for sub_file in os.listdir(os.path.join(DATA_PATH, file)):
                if not sub_file.endswith(".zip"):
                    for signal in os.listdir(os.path.join(DATA_PATH, file, sub_file)):
                        if signal in desired_signals:
                            df = pd.read_csv(os.path.join(DATA_PATH, file, sub_file, signal), names=names[signal],
                                             header=None)
                            if not df.empty:
                                if signal == 'ACC.csv':
                                    acc = pd.concat([acc, self.process_df(df, file)])
                                if signal == 'EDA.csv':
                                    eda = pd.concat([eda, self.process_df(df, file)])
                                if signal == 'HR.csv':
                                    hr = pd.concat([hr, self.process_df(df, file)])
                                if signal == 'TEMP.csv':
                                    temp = pd.concat([temp, self.process_df(df, file)])
                                if signal == 'BVP.csv':
                                    bvp = pd.concat([bvp, self.process_df(df, file)])
                                if signal == 'IBI.csv':
                                    ibi = pd.concat([ibi, self.process_df(df, file)])

        print("Saving data!")
        acc.to_csv(os.path.join(self.COMBINED_DATA_SAVE_LOCAL, 'combined_acc.csv'), index=False)
        eda.to_csv(os.path.join(self.COMBINED_DATA_SAVE_LOCAL, 'combined_eda.csv'), index=False)
        hr.to_csv(os.path.join(self.COMBINED_DATA_SAVE_LOCAL, 'combined_hr.csv'), index=False)
        temp.to_csv(os.path.join(self.COMBINED_DATA_SAVE_LOCAL, 'combined_temp.csv'), index=False)
        bvp.to_csv(os.path.join(self.COMBINED_DATA_SAVE_LOCAL, 'combined_bvp.csv'), index=False)
        ibi.to_csv(os.path.join(self.COMBINED_DATA_SAVE_LOCAL, 'combined_ibi.csv'), index=False)

    def merge_data(self):
        try:
            os.mkdir(self.MERGED_DATA_SAVE_LOCAL)
        except:
            pass

        signals = self.SIGNALS

        acc = pd.read_csv(os.path.join(self.COMBINED_DATA_SAVE_LOCAL, f"combined_{'acc'}.csv"), dtype={'id': str})
        eda = pd.read_csv(os.path.join(self.COMBINED_DATA_SAVE_LOCAL, f"combined_{'eda'}.csv"), dtype={'id': str})
        hr = pd.read_csv(os.path.join(self.COMBINED_DATA_SAVE_LOCAL, f"combined_{'hr'}.csv"), dtype={'id': str})
        temp = pd.read_csv(os.path.join(self.COMBINED_DATA_SAVE_LOCAL, f"combined_{'temp'}.csv"), dtype={'id': str})
        bvp = pd.read_csv(os.path.join(self.COMBINED_DATA_SAVE_LOCAL, f"combined_{'bvp'}.csv"), dtype={'id': str})
        ibi = pd.read_csv(os.path.join(self.COMBINED_DATA_SAVE_LOCAL, f"combined_{'ibi'}.csv"), dtype={'id': str})

        ids = eda['id'].unique()

        results = []
        for id in ids:
            results.append(self.merge_parallel(id, acc, eda, hr, temp, bvp, ibi))

        new_df = pd.concat(results, ignore_index=True)

        print("Saving data!")
        new_df.to_csv(os.path.join(self.MERGED_DATA_SAVE_LOCAL, "merged_data.csv"), index=False)

    def label_data(self):
        # Read Files
        PATH = self.CONTENT_ROOT + self.MERGED_DATA_SAVE_LOCAL

        df = pd.read_csv(os.path.join(PATH, 'merged_data.csv'), dtype={'id': str})
        df['datetime'] = pd.to_datetime(df['datetime'].apply(lambda x: x * (10 ** 9)))

        survey_path = self.SURVEY_PATH

        survey_df = pd.read_excel(survey_path, usecols=['ID', 'Start time', 'End time', 'date', 'Stress level'],
                                  dtype={'ID': str})
        survey_df['Stress level'].replace('na', np.nan, inplace=True)
        survey_df.dropna(inplace=True)

        survey_df['Start datetime'] = pd.to_datetime(
            survey_df['date'].map(str) + ' ' + survey_df['Start time'].map(str))
        survey_df['End datetime'] = pd.to_datetime(survey_df['date'].map(str) + ' ' + survey_df['End time'].map(str))
        survey_df.drop(['Start time', 'End time', 'date'], axis=1, inplace=True)

        # Convert SurveyResults.xlsx to GMT-00:00
        print("Converting ...")
        daylight = pd.to_datetime(datetime(2020, 11, 1, 0, 0))

        survey_df1 = survey_df[survey_df['End datetime'] <= daylight].copy()
        survey_df1['Start datetime'] = survey_df1['Start datetime'].apply(lambda x: x + timedelta(hours=5))
        survey_df1['End datetime'] = survey_df1['End datetime'].apply(lambda x: x + timedelta(hours=5))

        survey_df2 = survey_df.loc[survey_df['End datetime'] > daylight].copy()
        survey_df2['Start datetime'] = survey_df2['Start datetime'].apply(lambda x: x + timedelta(hours=6))
        survey_df2['End datetime'] = survey_df2['End datetime'].apply(lambda x: x + timedelta(hours=6))

        survey_df = pd.concat([survey_df1, survey_df2], ignore_index=True)
        # survey_df = survey_df.loc[survey_df['Stress level'] != 1.0]

        survey_df.reset_index(drop=True, inplace=True)

        # Label Data
        ids = df['id'].unique()

        results = []
        for id in ids:
            results.append(self.parallel(id, df, survey_df))

        new_df = pd.concat(results, ignore_index=True)

        print("Saving data!")
        new_df.to_csv(os.path.join(PATH, 'merged_data_labeled.csv'), index=False)


if __name__ == "__main__":
    pp = Loader()
    pp.CONTENT_ROOT = ''  # set this to the directory of the dataset
    pp.reset_dataset()
