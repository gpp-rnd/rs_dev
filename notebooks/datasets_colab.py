import pandas as pd
from sklearn.model_selection import train_test_split, GroupShuffleSplit

class GuideDataset:
    """Parent class for datasets for modeling sgRNA activity
    """
    def __init__(self, filepath, sgrna_seq_col, context_seq_col, rank_col, name,
                 random_seed=7, sgrna_group_col=None, cut_perc_col=None, classtype='dataset'):
        """Initialize class object

        :param filepath: str, path to activity data
        :param rank_col: str, column for ranking sgRNAs
        :param context_seq_col: str, column with context sequence
        :param sgrna_seq_col: str, column with sgrna sequence
        :param name: str, name of dataset
        :param sgrna_group_col: str or None, column to group guides by (e.g. gene)
        :param cut_perc_col: str or None, column indicating where within a protein an sgRNA cuts
        """
        self.filepath = filepath
        self.sgrna_seq_col = sgrna_seq_col
        self.context_seq_col = context_seq_col
        self.rank_col = rank_col
        self.sgrna_group_col = sgrna_group_col
        self.cut_perc_col = cut_perc_col
        self.random_seed = random_seed
        self.name = name
        self.dataset = None
        self.sgrnas = None
        self.classtype = classtype

    def load_data(self):
        if '.csv' in self.filepath:
            self.dataset = pd.read_csv(self.filepath)
        else:
            raise ValueError('Please save data as a csv')
    
    def check_data_loaded(self):
        if self.dataset is None:
            raise ValueError('Dataset must be loaded')
    
    def get_sg_df(self, sg_name='sgRNA Sequence', context_name='sgRNA Context Sequence', 
                  pam_name='PAM Sequence', include_group=False, group_name='sgRNA Target',
                  include_activity=False, activity_name='sgRNA Activity'):
        self.check_data_loaded()
        sg_df = (self.dataset[[self.sgrna_seq_col, self.context_seq_col]]
                 .rename({self.sgrna_seq_col: sg_name, self.context_seq_col: context_name}, axis=1)
                 .drop_duplicates())
        sg_df[pam_name] = sg_df[context_name].str[-6:-3]
        if include_group & (self.sgrna_group_col is not None):
            sg_df[group_name] = self.dataset[self.sgrna_group_col]
        if include_activity:
            sg_df[activity_name] = self.dataset[self.rank_col]
        return sg_df

    def get_sgrnas(self):
        self.sgrnas = set(self.dataset[self.sgrna_seq_col].to_list())

    def split_validation(self, val_frac=0.2, prepicked_sgs=None, random_state=7):
        if prepicked_sgs is not None:
            train_df = self.dataset[~self.dataset[self.sgrna_seq_col].isin(prepicked_sgs)]
            val_df = self.dataset[self.dataset[self.sgrna_seq_col].isin(prepicked_sgs)]
        else:
            train_df = self.dataset
            val_df = self.dataset.iloc[:0, :]  # empty dataframe
        curr_val_frac = val_df.shape[0]/(train_df.shape[0] + val_df.shape[0])
        if curr_val_frac < val_frac:
            add_val_frac = val_frac - curr_val_frac
            if self.sgrna_group_col is None:
                train_df, new_val_df = train_test_split(train_df, test_size=add_val_frac, random_state=random_state)
            else:
                train_df, new_val_df = next(GroupShuffleSplit(n_splits=1, test_size=add_val_frac,
                                                              random_state=random_state)
                                            .split(train_df, groups=train_df[self.sgrna_group_col]))
            val_df = pd.concat([val_df, new_val_df])
        return train_df.reset_index(drop=True), val_df.reset_index(drop=True)



aguirre_data = GuideDataset(filepath='Aguirre2017_activity.csv',
                            sgrna_seq_col='sgRNA Sequence', context_seq_col='sgRNA Context Sequence',
                            rank_col='avg_rank', name='Aguirre2017',
                            sgrna_group_col='Target Gene Symbol', cut_perc_col='Target Cut %')

chari_data = GuideDataset(filepath='Chari2015_activity.csv',
                          sgrna_seq_col='sgRNA Sequence', context_seq_col='sgRNA Context Sequence',
                          rank_col='log10(293T mutation rate)', name='Chari2015')

deweirdt_data = GuideDataset(filepath='DeWeirdt2020_activity.csv',
                             sgrna_seq_col='sgRNA Sequence', context_seq_col='sgRNA Context Sequence',
                             rank_col='avg_rank', name='DeWeirdt2020',
                             sgrna_group_col='Target Gene Symbol', cut_perc_col='Target Cut %')

doench2014_mouse_data = GuideDataset(filepath='Doench2014_mouse_activity.csv',
                                     sgrna_seq_col='sgRNA Sequence', context_seq_col='sgRNA Context Sequence',
                                     rank_col='gene_rank', name='Doench2014_mouse',
                                     sgrna_group_col='Target Gene Symbol', cut_perc_col='Target Cut %')

doench2014_human_data = GuideDataset(filepath='Doench2014_human_activity.csv',
                                     sgrna_seq_col='sgRNA Sequence', context_seq_col='sgRNA Context Sequence',
                                     rank_col='avg_rank', name='Doench2014_human',
                                     sgrna_group_col='Target Gene Symbol', cut_perc_col='Target Cut %')

doench2016_data = GuideDataset(filepath='Doench2016_activity.csv',
                               sgrna_seq_col='sgRNA Sequence', context_seq_col='sgRNA Context Sequence',
                               rank_col='avg_rank', name='Doench2016',
                               sgrna_group_col='Target Gene Symbol', cut_perc_col='Target Cut %')

kim_train_data = GuideDataset(filepath='Kim2019_train_activity.csv',
                              sgrna_seq_col='sgRNA Sequence', context_seq_col='sgRNA Context Sequence',
                              rank_col='indel_freq', name='Kim2019_train')

kim_test_data = GuideDataset(filepath='Kim2019_test_activity.csv',
                             sgrna_seq_col='sgRNA Sequence', context_seq_col='sgRNA Context Sequence',
                             rank_col='indel_freq', name='Kim2019_test')

koike_data = GuideDataset(filepath='Koike-Yusa2014_activity.csv',
                          sgrna_seq_col='sgRNA Sequence', context_seq_col='sgRNA Context Sequence',
                          rank_col='gene_rank', name='Koike-Yusa2014',
                          sgrna_group_col='Target Gene Symbol', cut_perc_col='Target Cut %')

shalem_data = GuideDataset(filepath='Shalem2014_activity.csv',
                           sgrna_seq_col='sgRNA Sequence', context_seq_col='sgRNA Context Sequence',
                           rank_col='gene_rank', name='Shalem_2014',
                           sgrna_group_col='Target Gene Symbol', cut_perc_col='Target Cut %')

wang_data = GuideDataset('Wang2014_activity.csv',
                         sgrna_seq_col='sgRNA Sequence', context_seq_col='sgRNA Context Sequence',
                         rank_col='avg_rank', name='Wang2014',
                         sgrna_group_col='Target Gene Symbol', cut_perc_col='Target Cut %')

dataset_list = [aguirre_data, chari_data, deweirdt_data, doench2014_mouse_data, doench2014_human_data,
                doench2016_data, kim_train_data, kim_test_data, koike_data, shalem_data, wang_data]




