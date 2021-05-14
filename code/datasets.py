import pandas as pd


class GuideDataset:
    """Parent class for datasets for modeling sgRNA activity
    """
    def __init__(self, filepath, sgrna_seq_col, context_seq_col, rank_col, name, endogenous,
                 random_seed=7, sgrna_group_col=None, cut_perc_col=None, classtype='dataset',
                 design_file=None):
        """Initialize class object

        :param filepath: str, path to activity data
        :param rank_col: str, column for ranking sgRNAs
        :param context_seq_col: str, column with context sequence
        :param sgrna_seq_col: str, column with sgrna sequence
        :param name: str, name of dataset
        :param endogenous: bool
        :param sgrna_group_col: str or None, column to group guides by (e.g. gene)
        :param cut_perc_col: str or None, column indicating where within a protein an sgRNA cuts
        :param classtype: name of class (#TODO - redundant to parent class)
        :param design_file: str, path to parquet file with all designs for target organism
        """
        self.filepath = filepath
        self.sgrna_seq_col = sgrna_seq_col
        self.context_seq_col = context_seq_col
        self.rank_col = rank_col
        self.endogenous = endogenous
        self.sgrna_group_col = sgrna_group_col
        self.cut_perc_col = cut_perc_col
        self.random_seed = random_seed
        self.name = name
        self.classtype = classtype
        self.design_file = design_file
        self.dataset = None
        self.sgrnas = None

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

    def set_sgrnas(self):
        self.sgrnas = set(self.dataset[self.sgrna_seq_col].to_list())

    def get_designs(self):
        if self.sgrnas is None:
            raise ValueError('Must load sgRNA sequence')
        sg_list = list(self.sgrnas)
        design_df = pd.read_parquet(
            self.design_file,
            filters=[[('sgRNA Sequence', 'in', sg_list)]])
        return design_df


mouse_designs = '/Volumes/GoogleDrive/Shared drives/GPP Cloud /R&D/People/Peter/gpp-annotation-files/sgRNA_design_10090_GRCm38_SpyoCas9_CRISPRko_Ensembl_20200406.parquet'
human_designs = '/Volumes/GoogleDrive/Shared drives/GPP Cloud /R&D/People/Peter/gpp-annotation-files/sgRNA_design_9606_GRCh38_SpyoCas9_CRISPRko_Ensembl_20200401.parquet'

aguirre_data = GuideDataset(filepath='../data/processed/Aguirre2017_activity.csv',
                            sgrna_seq_col='sgRNA Sequence', context_seq_col='sgRNA Context Sequence',
                            rank_col='avg_rank', endogenous=True, name='Aguirre2017',
                            sgrna_group_col='Target Gene Symbol', cut_perc_col='Target Cut %',
                            design_file=human_designs)

chari_data = GuideDataset(filepath='../data/processed/Chari2015_activity.csv',
                          sgrna_seq_col='sgRNA Sequence', context_seq_col='sgRNA Context Sequence',
                          rank_col='log10(293T mutation rate)', endogenous=False, name='Chari2015')

deweirdt_data = GuideDataset(filepath='../data/processed/DeWeirdt2020_activity.csv',
                             sgrna_seq_col='sgRNA Sequence', context_seq_col='sgRNA Context Sequence',
                             rank_col='avg_rank', endogenous=True, name='DeWeirdt2020',
                             sgrna_group_col='Target Gene Symbol', cut_perc_col='Target Cut %',
                             design_file=human_designs)

doench2014_mouse_data = GuideDataset(filepath='../data/processed/Doench2014_mouse_activity.csv',
                                     sgrna_seq_col='sgRNA Sequence', context_seq_col='sgRNA Context Sequence',
                                     rank_col='gene_rank', endogenous=True, name='Doench2014_mouse',
                                     sgrna_group_col='Target Gene Symbol', cut_perc_col='Target Cut %',
                                     design_file=mouse_designs)

doench2014_human_data = GuideDataset(filepath='../data/processed/Doench2014_human_activity.csv',
                                     sgrna_seq_col='sgRNA Sequence', context_seq_col='sgRNA Context Sequence',
                                     rank_col='avg_rank', endogenous=True, name='Doench2014_human',
                                     sgrna_group_col='Target Gene Symbol', cut_perc_col='Target Cut %',
                                     design_file=human_designs)

doench2016_data = GuideDataset(filepath='../data/processed/Doench2016_activity.csv',
                               sgrna_seq_col='sgRNA Sequence', context_seq_col='sgRNA Context Sequence',
                               rank_col='avg_rank', endogenous=True, name='Doench2016',
                               sgrna_group_col='Target Gene Symbol', cut_perc_col='Target Cut %',
                               design_file=human_designs)

kim_train_data = GuideDataset(filepath='../data/processed/Kim2019_train_activity.csv',
                              sgrna_seq_col='sgRNA Sequence', context_seq_col='sgRNA Context Sequence',
                              rank_col='indel_freq', endogenous=False, name='Kim2019_train')

kim_test_data = GuideDataset(filepath='../data/processed/Kim2019_test_activity.csv',
                             sgrna_seq_col='sgRNA Sequence', context_seq_col='sgRNA Context Sequence',
                             rank_col='indel_freq', endogenous=False, name='Kim2019_test')

koike_data = GuideDataset(filepath='../data/processed/Koike-Yusa2014_activity.csv',
                          sgrna_seq_col='sgRNA Sequence', context_seq_col='sgRNA Context Sequence',
                          rank_col='gene_rank', endogenous=True, name='Koike-Yusa2014',
                          sgrna_group_col='Target Gene Symbol', cut_perc_col='Target Cut %',
                          design_file=mouse_designs)

shalem_data = GuideDataset(filepath='../data/processed/Shalem2014_activity.csv',
                           sgrna_seq_col='sgRNA Sequence', context_seq_col='sgRNA Context Sequence',
                           rank_col='gene_rank', endogenous=True, name='Shalem_2014',
                           sgrna_group_col='Target Gene Symbol', cut_perc_col='Target Cut %',
                           design_file=human_designs)

wang_data = GuideDataset('../data/processed/Wang2014_activity.csv',
                         sgrna_seq_col='sgRNA Sequence', context_seq_col='sgRNA Context Sequence',
                         rank_col='avg_rank', endogenous=True, name='Wang2014',
                         sgrna_group_col='Target Gene Symbol', cut_perc_col='Target Cut %',
                         design_file=human_designs)

dataset_list = [aguirre_data, chari_data, deweirdt_data, doench2014_mouse_data, doench2014_human_data,
                doench2016_data, kim_train_data, kim_test_data, koike_data, shalem_data, wang_data]


def get_sg_groups_df(datasets):
    """Get DataFrame of sgRNAs with designs, activity, and target columns

    :param datasets: list of GuideDataset
    :return: DataFrame
    """
    for ds in datasets:
        ds.load_data()
        ds.set_sgrnas()
    sg_df_list = []
    for ds in datasets:
        sg_df = ds.get_sg_df(include_group=True, include_activity=True)
        sg_df['dataset'] = ds.name
        design_df = ds.get_designs()
        sg_df = sg_df.merge(design_df, how='inner',
                            on=['sgRNA Sequence', 'sgRNA Context Sequence', 'PAM Sequence'])
        sg_df_list.append(sg_df)
    sg_df_groups = (pd.concat(sg_df_list)
                    .groupby(['sgRNA Context Sequence'])
                    .agg(n_conditions=('sgRNA Context Sequence', 'count'),
                         target=('sgRNA Target', lambda x: ', '.join(set([s.upper() for s in x if not pd.isna(s)]))))
                    .reset_index())
    multi_target = sg_df_groups['target'].str.contains(',').sum()
    print('Context sequences with multiple targets: ' + str(multi_target))
    # handle singleton case
    sg_df_groups['target'] = sg_df_groups.apply(lambda row:
                                                row['target'] if (row['target'] != '') else
                                                row['sgRNA Context Sequence'],
                                                axis=1)
    # Note that 'target' is not in the sg_df_list, and is coming from the sg_df_groups df
    sg_df_class_groups = (pd.concat(sg_df_list)
                          .merge(sg_df_groups, how='inner', on='sgRNA Context Sequence')
                          .sort_values(['dataset', 'target'])
                          .reset_index(drop=True))
    return sg_df_class_groups


