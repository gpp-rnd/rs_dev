import target_featurization as ft
import pandas as pd
from datasets import dataset_list
import pytest
import numpy as np


@pytest.fixture
def sg_designs_endog():
    train_data_list = list()
    for ds in dataset_list:
        if ds.endogenous:
            train_data_list.append(ds)
    for ds in dataset_list:
        ds.load_data()
        ds.set_sgrnas()
    sg_df_list = []
    for ds in train_data_list:
        sg_df = ds.get_sg_df(include_group=True, include_activity=True)
        sg_df['dataset'] = ds.name
        design_df = ds.get_designs()
        sg_df = sg_df.merge(design_df, how='inner',
                            on=['sgRNA Sequence', 'sgRNA Context Sequence', 'PAM Sequence'])
        sg_df_list.append(sg_df)
    sg_df = (pd.concat(sg_df_list)
             .reset_index(drop=True))
    short_sg_df = sg_df.sample(100, random_state=1)
    new_designs = ft.add_target_columns(short_sg_df)
    return new_designs


@pytest.fixture
def aa_seq_df(sg_designs_endog):
    aa_seqs = pd.read_csv('../data/interim/aa_seqs.csv')
    filtered_aa_seqs = aa_seqs[aa_seqs['Transcript Base'].isin(sg_designs_endog['Transcript Base'])]
    return filtered_aa_seqs


@pytest.fixture
def codon_map():
    codon_map_df = pd.read_csv('../data/external/codon_map.csv')
    codon_map = pd.Series(codon_map_df['Amino Acid'].values, index=codon_map_df['Codon']).to_dict()
    return codon_map


@pytest.fixture
def domain_data(sg_designs_endog):
    protein_domains = pd.read_csv('../data/interim/protein_domains.csv')
    # TODO - move to database query step
    protein_domains = protein_domains.rename({'Parent': 'Transcript Base'}, axis=1)
    filtered_domains = protein_domains[protein_domains['Transcript Base'].isin(sg_designs_endog['Transcript Base'])]
    return filtered_domains


@pytest.fixture
def conservation_data(sg_designs_endog):
    transcript_bases = list(sg_designs_endog['Transcript Base'].unique())
    conservation_df = (pd.read_parquet('../data/interim/conservation.parquet',
                                       filters=[[('Transcript Base', 'in', transcript_bases)]])
                       .reset_index(drop=True))
    return conservation_df


def test_sg_designs_endog(sg_designs_endog):
    np.testing.assert_allclose(sg_designs_endog['AA Index']*3, sg_designs_endog['Target Cut Length'], atol=3)


def test_aa_seq_featurization():
    aas = ['A', 'C', 'D', 'E', 'F',
           'G', 'H', 'I', 'K', 'L',
           'M', 'N', 'P', 'Q', 'R',
           'S', 'T', 'V', 'W', 'Y', '*']
    target = 'ACDG*-'
    sequence_order = ['-2', '-1', '0', '1', '2', '3']
    ft_dict = {}
    ft.get_one_aa_frac(ft_dict, target, aas)
    assert ft_dict['A'] == 1/6
    assert ft_dict['Q'] == 0
    ft.get_two_aa_frac(ft_dict, target, aas)
    assert ft_dict['DG'] == 1/5
    ft.get_one_aa_pos(ft_dict, target, aas, sequence_order)
    assert ft_dict['-1C'] == 1
    ft_dict_df = ft.featurize_aa_seqs(pd.Series([target, 'CDG*--', 'LLLLLL']))
    assert ft_dict_df.loc['LLLLLL', 'Hydrophobicity'] == ft_dict_df['Hydrophobicity'].max()


def get_rev_comp(sgrna):
    """Get reverse compliment of a guide"""
    nt_map = {'A': 'T', 'T': 'A', 'G': 'C', 'C': 'G'}
    rev_comp = ''
    for nt in sgrna:
        rev_comp += nt_map[nt]
    rev_comp = rev_comp[::-1]
    return rev_comp


def test_aa_features(sg_designs_endog, aa_seq_df, codon_map):
    aa_features = ft.get_amino_acid_features(sg_designs_endog, aa_seq_df, width=10,
                                             features=['Pos. Ind. 1mer', 'Pos. Ind. 2mer', 'Pos. Dep. 1mer',
                                                       'Hydrophobicity', 'Aromaticity',
                                                       'Isoelectric Point', 'Secondary Structure'],
                                             id_cols=['sgRNA Context Sequence', 'Target Cut Length',
                                                      'Target Transcript', 'Orientation'])
    assert (aa_features['AA Subsequence'].str.len() == 20).all()
    row = aa_features.sample(1, random_state=7).iloc[0, :]
    subseq = row['AA Subsequence']
    assert row['0' + subseq[9]] == 1
    context = row['sgRNA Context Sequence']
    rc_context = get_rev_comp(context)
    translations = dict()
    rc_translations = dict()
    for i in [0, 1, 2]:
        translations[i] = ''.join([codon_map[context[j:j+3]] for j in range(i, len(context), 3)
                                   if (j + 3) <= len(context)])
        rc_translations[i] = ''.join([codon_map[rc_context[j:j+3]] for j in range(i, len(rc_context), 3)
                                      if (j + 3) <= len(rc_context)])
    assert ((translations[0] in subseq) or (translations[1] in subseq) or (translations[2] in subseq) or
            (rc_translations[0] in subseq) or (rc_translations[1] in subseq) or (rc_translations[2] in subseq))


def test_domain_conservation(sg_designs_endog, domain_data, conservation_data):
    protein_domain_features = ft.get_protein_domain_features(sg_designs_endog, domain_data, sources=None,
                                                             id_cols=['sgRNA Context Sequence', 'Target Cut Length',
                                                                      'Target Transcript', 'Orientation'])
    conservation_features = ft.get_conservation_features(sg_designs_endog, conservation_data,
                                                         small_width=6, large_width=50,
                                                         conservation_column='ranked_conservation',
                                                         id_cols=['sgRNA Context Sequence', 'Target Cut Length',
                                                                  'Target Transcript', 'Orientation'])
    merged_features = protein_domain_features.merge(conservation_features, how='inner', on=['sgRNA Context Sequence',
                                                                                            'Target Cut Length',
                                                                                            'Target Transcript',
                                                                                            'Orientation'])
    smart_avg_cons = merged_features.loc[merged_features['Smart'].astype(bool), 'cons_12'].mean()
    non_smart_avg_cons = merged_features.loc[~merged_features['Smart'].astype(bool), 'cons_12'].mean()
    assert smart_avg_cons > non_smart_avg_cons


def test_featurization(sg_designs_endog, aa_seq_df, domain_data, conservation_data):
    feature_df, feature_list = ft.build_target_feature_df(sg_designs=sg_designs_endog,
                                                          aa_seq_df=aa_seq_df,
                                                          protein_domain_df=domain_data,
                                                          conservation_df=conservation_data)
    assert len(feature_list) > 40
    assert feature_df.shape[0] == sg_designs_endog.shape[0]
