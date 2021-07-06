import pandas as pd


class GuideDataset:
    """Parent class for datasets for modeling sgRNA activity
    """
    def __init__(self, filepath, sgrna_seq_col, context_seq_col, rank_col, name, endogenous, genomewide,
                 random_seed=7, sgrna_group_col=None, cut_perc_col=None, classtype='dataset',
                 design_file=None, tracr='Hsu2013'):
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
        self.genomewide = genomewide
        self.sgrna_group_col = sgrna_group_col
        self.cut_perc_col = cut_perc_col
        self.random_seed = random_seed
        self.name = name
        self.classtype = classtype
        self.design_file = design_file
        self.dataset = None
        self.sgrnas = None
        self.tracr = tracr

    def load_data(self):
        if self.dataset is None:
            if '.csv' in self.filepath:
                self.dataset = pd.read_csv(self.filepath)
            elif '.txt' in self.filepath:
                self.dataset = pd.read_table(self.filepath)
            else:
                raise ValueError('Please save data as a csv')
        # else dataset already loaded
    
    def check_data_loaded(self):
        if self.dataset is None:
            raise ValueError('Dataset must be loaded')
    
    def get_sg_df(self, sg_name='sgRNA Sequence', context_name='sgRNA Context Sequence', 
                  include_pam=True, pam_name='PAM Sequence',
                  include_group=False, group_name='sgRNA Target',
                  include_activity=False, activity_name='sgRNA Activity'):
        self.check_data_loaded()
        sg_df = (self.dataset[[self.sgrna_seq_col, self.context_seq_col]]
                 .rename({self.sgrna_seq_col: sg_name, self.context_seq_col: context_name}, axis=1))
        if include_pam:
            # PAM for SpCas9
            sg_df[pam_name] = sg_df[context_name].str[-6:-3]
        if include_group & (self.sgrna_group_col is not None):
            sg_df[group_name] = self.dataset[self.sgrna_group_col]
        if include_activity:
            sg_df[activity_name] = self.dataset[self.rank_col]
        sg_df = sg_df.drop_duplicates()
        return sg_df

    def set_sgrnas(self):
        if self.sgrnas is None:
            self.sgrnas = set(self.dataset[self.sgrna_seq_col].to_list())
        # else sgRNAs already loaded

    def get_designs(self):
        if self.sgrnas is None:
            raise ValueError('Must load sgRNA sequence')
        sg_list = list(self.sgrnas)
        design_df = pd.read_parquet(
            self.design_file,
            filters=[[('sgRNA Sequence', 'in', sg_list)]])
        return design_df


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
        sg_df['tracr'] = ds.tracr
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


mouse_designs = '/Volumes/GoogleDrive/Shared drives/GPP Cloud /R&D/People/Peter/gpp-annotation-files/sgRNA_design_10090_GRCm38_SpyoCas9_CRISPRko_Ensembl_20200406.parquet'
human_designs = '/Volumes/GoogleDrive/Shared drives/GPP Cloud /R&D/People/Peter/gpp-annotation-files/sgRNA_design_9606_GRCh38_SpyoCas9_CRISPRko_Ensembl_20200401.parquet'

aguirre_data = GuideDataset(filepath='../data/processed/Aguirre2017_activity.csv',
                            sgrna_seq_col='sgRNA Sequence', context_seq_col='sgRNA Context Sequence',
                            rank_col='avg_mean_centered_neg_lfc', endogenous=True, name='Aguirre2017',
                            sgrna_group_col='Target Gene Symbol', cut_perc_col='Target Cut %',
                            design_file=human_designs, genomewide=True)

chari_data = GuideDataset(filepath='../data/processed/Chari2015_activity.csv',
                          sgrna_seq_col='sgRNA Sequence', context_seq_col='sgRNA Context Sequence',
                          rank_col='scaled_mutation_rate', endogenous=False, name='Chari2015',
                          genomewide=False)

doench2014_mouse_data = GuideDataset(filepath='../data/processed/Doench2014_mouse_activity.csv',
                                     sgrna_seq_col='sgRNA Sequence', context_seq_col='sgRNA Context Sequence',
                                     rank_col='scaled_lfc', endogenous=True, name='Doench2014_mouse',
                                     sgrna_group_col='Target Gene Symbol', cut_perc_col='Target Cut %',
                                     design_file=mouse_designs, genomewide=False)

doench2014_human_data = GuideDataset(filepath='../data/processed/Doench2014_human_activity.csv',
                                     sgrna_seq_col='sgRNA Sequence', context_seq_col='sgRNA Context Sequence',
                                     rank_col='scaled_lfc', endogenous=True, name='Doench2014_human',
                                     sgrna_group_col='Target Gene Symbol', cut_perc_col='Target Cut %',
                                     design_file=human_designs, genomewide=False)

doench2016_data = GuideDataset(filepath='../data/processed/Doench2016_activity.csv',
                               sgrna_seq_col='sgRNA Sequence', context_seq_col='sgRNA Context Sequence',
                               rank_col='scaled_lfc', endogenous=True, name='Doench2016',
                               sgrna_group_col='Target Gene Symbol', cut_perc_col='Target Cut %',
                               design_file=human_designs, genomewide=False)

kim_train_data = GuideDataset(filepath='../data/processed/Kim2019_train_activity.csv',
                              sgrna_seq_col='sgRNA Sequence', context_seq_col='sgRNA Context Sequence',
                              rank_col='scaled_indels', endogenous=False, name='Kim2019_train',
                              genomewide=False)

kim_test_data = GuideDataset(filepath='../data/processed/Kim2019_test_activity.csv',
                             sgrna_seq_col='sgRNA Sequence', context_seq_col='sgRNA Context Sequence',
                             rank_col='scaled_indels', endogenous=False, name='Kim2019_test',
                             genomewide=False)

koike_data = GuideDataset(filepath='../data/processed/Koike-Yusa2014_activity.csv',
                          sgrna_seq_col='sgRNA Sequence', context_seq_col='sgRNA Context Sequence',
                          rank_col='mean_centered_neg_lfc', endogenous=True, name='Koike-Yusa2014',
                          sgrna_group_col='Target Gene Symbol', cut_perc_col='Target Cut %',
                          design_file=mouse_designs, genomewide=True)

shalem_data = GuideDataset(filepath='../data/processed/Shalem2014_activity.csv',
                           sgrna_seq_col='sgRNA Sequence', context_seq_col='sgRNA Context Sequence',
                           rank_col='mean_centered_neg_lfc', endogenous=True, name='Shalem_2014',
                           sgrna_group_col='Target Gene Symbol', cut_perc_col='Target Cut %',
                           design_file=human_designs, genomewide=True)

wang_data = GuideDataset('../data/processed/Wang2014_activity.csv',
                         sgrna_seq_col='sgRNA Sequence', context_seq_col='sgRNA Context Sequence',
                         rank_col='scaled_neg_lfc', endogenous=True, name='Wang2014',
                         sgrna_group_col='Target Gene Symbol', cut_perc_col='Target Cut %',
                         design_file=human_designs, genomewide=False)

xiang_data = GuideDataset(filepath='../data/processed/Xiang2021_activity.csv',
                          sgrna_seq_col='sgRNA Sequence', context_seq_col='sgRNA Context Sequence',
                          rank_col='scaled_indel_eff', endogenous=False, name='Xiang2021',
                          design_file=human_designs, genomewide=False)

behan_data = GuideDataset(filepath='../data/processed/Behan2019_activity.csv',
                          sgrna_seq_col='sgRNA Sequence', context_seq_col='sgRNA Context Sequence',
                          rank_col='avg_mean_centered_neg_lfc', endogenous=True, name='Behan2019',
                          sgrna_group_col='Target Gene Symbol', cut_perc_col='Target Cut %',
                          design_file=human_designs, genomewide=True, tracr='Chen2013')

munoz_data = GuideDataset(filepath='../data/processed/Munoz2016_activity.csv',
                          sgrna_seq_col='sgRNA Sequence', context_seq_col='sgRNA Context Sequence',
                          rank_col='gene_scaled_neg_zscore', endogenous=True, name='Munoz2016',
                          sgrna_group_col='Target Gene Symbol', cut_perc_col='Target Cut %',
                          design_file=human_designs, genomewide=False, tracr='Chen2013')

dataset_list = [aguirre_data, chari_data, doench2014_mouse_data, doench2014_human_data,
                doench2016_data, kim_train_data, kim_test_data, koike_data, shalem_data,
                wang_data, xiang_data, behan_data, munoz_data]

tracr_list = ['Hsu2013', 'Chen2013']

sa_designs = '/Volumes/GoogleDrive/Shared drives/GPP Cloud /R&D/People/Peter/gpp-annotation-files/sgRNA_design_9606_GRCh38_SaurCas9_CRISPRko_Ensembl_20200401.parquet'
encas12a_designs = '/Volumes/GoogleDrive/Shared drives/GPP Cloud /R&D/People/Peter/gpp-annotation-files/sgRNA_design_9606_GRCh38_enAsCas12a_CRISPRko_Ensembl_20200401.parquet'

doench2018_sa = GuideDataset(filepath='../data/external/Supplementary Table 1 Saureus model input.txt',
                             sgrna_seq_col='Construct Barcode', context_seq_col='30mer',
                             rank_col='rank', endogenous=True, name='Doench2018_SaCas9',
                             sgrna_group_col='Target gene', cut_perc_col='Pct Pep',
                             design_file=sa_designs, genomewide=False)

deweirdt2020_encas12a = GuideDataset(filepath='../data/external/2019-12-12_encas12a_pam_tiling_train.csv',
                                     sgrna_seq_col='Construct Barcode', context_seq_col='Context Sequence',
                                     rank_col='activity_rank', endogenous=True, name='DeWeirdt2020_enCas12a',
                                     sgrna_group_col='Gene Symbol', design_file=encas12a_designs,
                                     genomewide=False)

external_dataset_list = [doench2018_sa, deweirdt2020_encas12a]
expanded_dataset_list = external_dataset_list + dataset_list

