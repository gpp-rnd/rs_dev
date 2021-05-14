import pandas as pd
from Bio.SeqUtils.ProtParam import ProteinAnalysis

# Updating Design Matrix


def add_target_columns(sg_df):
    """Add ['AA Index' and 'Transcript Base'] to design df

    :param sg_df: DataFrame
    :return: DataFrame
    """
    out_df = sg_df.copy()
    out_df['AA Index'] = (out_df['Target Cut Length'] - 1) // 3 + 1
    out_df['Transcript Base'] = out_df['Target Transcript'].str.split('.', expand=True)[0]
    return out_df


# Position Features


def get_position_features(sg_df):
    """Get  features ['dist from start', 'dist from end', 'dist percent', 'sense']

    :param sg_df: DataFrame
    :return: DataFrame
    """
    position_df = sg_df[['sgRNA Context Sequence', 'Transcript Base', 'Target Cut %']].copy()
    position_df['sense'] = sg_df['Orientation'] == 'sense'
    return position_df


# Amino Acid Features


def get_one_aa_pos(feature_dict, aa_sequence, aas, sequence_order):
    """One hot encode single amino acids

    :param feature_dict: dict, feature dictionary
    :param aa_sequence: str, amino acid sequence
    :param aas: list, list of amino acids
    :param sequence_order: list, position mapping for sequence
    """
    for i, pos in enumerate(sequence_order):
        curr_nt = aa_sequence[i]
        for aa in aas:
            key = pos + aa
            if curr_nt == aa:
                feature_dict[key] = 1
            else:
                feature_dict[key] = 0


def get_one_aa_frac(feature_dict, aa_sequence, aas):
    """Get fraction of single aa

    :param feature_dict: dict, feature dictionary
    :param aa_sequence: str, amino acid sequence
    :param aas: list, list of amino acids
    """
    for aa in aas:
        aa_frac = aa_sequence.count(aa) / len(aa_sequence)
        feature_dict[aa] = aa_frac


def get_two_aa_frac(feature_dict, aa_sequence, aas):
    """Get fraction of double aas

    :param feature_dict: dict, feature dictionary
    :param aa_sequence: str, amino acid sequence
    :param aas: list, list of amino acids
    """
    for aa1 in aas:
        for aa2 in aas:
            aa_frac = aa_sequence.count(aa1 + aa2) / (len(aa_sequence) - 1)
            feature_dict[aa1 + aa2] = aa_frac


def get_aa_aromaticity(feature_dict, analyzed_seq):
    """Get fraction of aromatic amino acids in a sequence.

    Phe (F) + Trp (W) + Tyr (Y)

    :param feature_dict:
    :param analyzed_seq: ProteinAnalysis object
    """
    feature_dict['Aromaticity'] = analyzed_seq.aromaticity()


def get_aa_hydrophobicity(feature_dict, analyzed_seq):
    """Grand Average of Hydropathy

     The GRAVY value is calculated by adding the hydropathy value for each residue and dividing
     by the length of the sequence (Kyte and Doolittle; 1982). The larger the number, the more hydrophobic the
     amino acid

    :param feature_dict: dict
    :param analyzed_seq: ProteinAnalysis object
    """
    feature_dict['Hydrophobicity'] = analyzed_seq.gravy()


def get_aa_ip(feature_dict, analyzed_seq):
    """Get the Isoelectric Point of an amino acid sequence

    Charge of amino acid

    :param feature_dict: dict
    :param analyzed_seq: ProteinAnalysis object
    """
    feature_dict['Isoelectric Point'] = analyzed_seq.isoelectric_point()


def get_aa_secondary_structure(feature_dict, analyzed_seq):
    """Get the fraction of amion acids that tend to be in a helix, turn or sheet

    :param feature_dict: dict
    :param analyzed_seq: ProteinAnalysis object
    """
    feature_dict['Helix'], feature_dict['Turn'], feature_dict['Sheet'] = analyzed_seq.secondary_structure_fraction()


def featurize_aa_seqs(aa_sequences, features=None):
    """Get feature DataFrame for a list of amino acid sequences

    :param aa_sequences: list of str
    :param features: list or None
    :return: DataFrame
    """
    if features is None:
        features = ['Pos. Ind. 1mer', 'Hydrophobicity', 'Aromaticity',
                    'Isoelectric Point',
                    'Secondary Structure']
    seq_len = len(aa_sequences[0])
    sequence_order = [str(int(x - seq_len/2 + 1)) for x in range(seq_len)]
    aas = ['A', 'C', 'D', 'E', 'F',
           'G', 'H', 'I', 'K', 'L',
           'M', 'N', 'P', 'Q', 'R',
           'S', 'T', 'V', 'W', 'Y', '*']
    clean_aa_seqs = aa_sequences.str.replace('\*|-', '', regex=True)
    feature_dict_list = []
    for i, (aa_sequence, clean_sequence) in enumerate(zip(aa_sequences, clean_aa_seqs)):
        analyzed_seq = ProteinAnalysis(clean_sequence)
        feature_dict = {}
        if 'Pos. Ind. 1mer' in features:
            get_one_aa_frac(feature_dict, aa_sequence, aas)
        if 'Pos. Ind. 2mer' in features:
            get_two_aa_frac(feature_dict, aa_sequence, aas)
        if 'Pos. Dep. 1mer' in features:
            get_one_aa_pos(feature_dict, aa_sequence, aas, sequence_order)
        if 'Hydrophobicity' in features:
            get_aa_hydrophobicity(feature_dict, analyzed_seq)
        if 'Aromaticity' in features:
            get_aa_aromaticity(feature_dict, analyzed_seq)
        if 'Isoelectric Point' in features:
            get_aa_ip(feature_dict, analyzed_seq)
        if 'Secondary Structure' in features:
            get_aa_secondary_structure(feature_dict, analyzed_seq)
        feature_dict_list.append(feature_dict)
    feature_matrix = pd.DataFrame(feature_dict_list)
    feature_matrix.index = aa_sequences
    return feature_matrix


def get_amino_acid_features(sg_designs, aa_seq_df, width=8, features=None):
    """Featurize amino acid sequences

    :param sg_designs: DataFrame
    :param aa_seq_df: DataFrame, Transcript Base and (AA) seq
    :param width: int, length on each side of the cut site
    :param feature: list
    :return: DataFrame
    """
    sg_aas = (aa_seq_df.merge(sg_designs[['Transcript Base', 'sgRNA Context Sequence', 'AA Index']],
                              how='inner',
                              on='Transcript Base'))
    padding = '-' * (width + 1)
    sg_aas['extended_seq'] = padding + sg_aas['seq'] + '*' + padding
    sg_aas['AA Index pad'] = sg_aas['AA Index'] + width + 1
    # One-indexed
    sg_aas['seq_start'] = sg_aas['AA Index pad'] - width
    sg_aas['seq_end'] = sg_aas['AA Index pad'] + width - 1
    # Zero-indexed for python
    sg_aas['AA Subsequence'] = sg_aas.apply(lambda row: row['extended_seq'][row['seq_start'] - 1:row['seq_end']],
                                            axis=1)
    aa_features = featurize_aa_seqs(sg_aas['AA Subsequence'], features=features)
    aa_features_annot = pd.concat([sg_aas[['Transcript Base', 'sgRNA Context Sequence', 'AA Subsequence']]
                                   .reset_index(drop=True),
                                   aa_features.reset_index(drop=True)], axis=1)
    return aa_features_annot


# Protein Domain


def get_protein_domain_features(sg_design_df, protein_domains, categories=None):
    """Get binary dataframe of protein domains

    :param sg_design_df: DataFrame, with columns ['Transcript Base', 'sgRNA Context Sequence', 'AA Index']
    :param protein_domains: DataFrame, with columns ['Transcript Base', 'type']
    :param categories: list. list of database types to include
    :return: DataFrame, with binary features for protein domains
    """
    if categories is None:
        categories = ['Pfam', 'PANTHER', 'HAMAP', 'SuperFamily', 'TIGRfam', 'ncoils', 'Gene3D',
                      'Prosite_patterns', 'Seg', 'SignalP', 'TMHMM', 'MobiDBLite',
                      'PIRSF', 'PRINTS', 'Smart', 'Prosite_profiles']
    protein_domains = protein_domains[protein_domains['type'].isin(categories)]
    domain_feature_df = sg_design_df[['Transcript Base', 'sgRNA Context Sequence', 'AA Index']].copy()
    domain_feature_df = domain_feature_df.merge(protein_domains,
                                                how='inner', on='Transcript Base')
    domain_feature_df = (domain_feature_df[domain_feature_df['AA Index'].between(domain_feature_df['start'],
                                                                                 domain_feature_df['end'])]
                         .copy())
    domain_feature_df = domain_feature_df[['Transcript Base', 'sgRNA Context Sequence', 'type']].drop_duplicates()
    domain_feature_df['present'] = 1
    domain_feature_df = (domain_feature_df.pivot_table(values='present',
                                                       index=['Transcript Base', 'sgRNA Context Sequence'],
                                                       columns='type',
                                                       fill_value=0)
                         .reset_index())
    # Ensure all domain columns are present for testing
    full_column_df = pd.DataFrame(columns=categories, dtype=int)  # empty
    domain_feature_df = pd.concat([full_column_df, domain_feature_df])
    return domain_feature_df


# Conservation


def get_conservation_ranges(cut_pos, small_width, large_width):
    small_range = range(cut_pos - small_width, cut_pos + small_width)
    large_range = range(cut_pos - large_width, cut_pos + large_width)
    return small_range, large_range


def get_conservation_features(sg_designs, conservation_df, conservation_column='ranked_conservation',
                              small_width=2, large_width=32):
    """Get conservation features

    :param sg_designs: DataFrame
    :param conservation_df: DataFrame, tidy conservation scores indexed by Transcript Base and target position
    :param conservation_column: str, name of column to calculate scores with
    :param small_width: int, small window length to average scores in one direction
    :param large_width: int, large window length to average scores in the one direction
    :return: DataFrame of conservation features
    """
    sg_designs_width = sg_designs[['sgRNA Context Sequence', 'Transcript Base', 'Target Cut Length']].copy()
    sg_designs_width['target position small'], sg_designs_width['target position large'] =  \
        zip(*sg_designs_width['Target Cut Length']
            .apply(get_conservation_ranges, small_width=small_width,
                   large_width=large_width))
    small_width_conservation = (sg_designs_width.drop('target position large', axis=1)
                                .rename({'target position small': 'target position'}, axis=1)
                                .explode('target position')
                                .merge(conservation_df, how='inner',
                                       on=['Transcript Base', 'target position'])
                                .groupby(['Transcript Base', 'sgRNA Context Sequence'])
                                .agg(cons=(conservation_column, 'mean'))
                                .rename({'cons': 'cons_' + str(small_width * 2)}, axis=1)
                                .reset_index())
    large_width_conservation = (sg_designs_width.drop('target position small', axis=1)
                                .rename({'target position large': 'target position'}, axis=1)
                                .explode('target position')
                                .merge(conservation_df, how='inner',
                                       on=['Transcript Base', 'target position'])
                                .groupby(['Transcript Base', 'sgRNA Context Sequence'])
                                .agg(cons=(conservation_column, 'mean'))
                                .rename({'cons': 'cons_' + str(large_width * 2)}, axis=1)
                                .reset_index())
    cons_feature_df = small_width_conservation.merge(large_width_conservation, how='outer',
                                                     on=['sgRNA Context Sequence', 'Transcript Base'])
    return cons_feature_df


def build_target_feature_df(sg_designs, features=None,
                            aa_seq_df=None, aa_width=8, aa_features=None,
                            protein_domain_df=None, protein_domain_sources=None,
                            conservation_df=None, conservation_column='ranked_conservation',
                            cons_small_width=2, cons_large_width=32):
    """Build the feature matrix for the sgRNA target site

    :param sg_designs: DataFrame
    :param features: list
    :param aa_seq_df: DataFrame
    :param aa_width: int
    :param aa_features: list
    :param protein_domain_df: DataFrame
    :param protein_domain_sources: list or None. Defaults to all sources except Sifts
    :param conservation_df: DataFrame
    :param conservation_column: str
    :param cons_small_width: int
    :param cons_large_width: int
    :return: (feature_df, feature_list)
        feature_df: DataFrame
        feature_list: list
    """
    if features is None:
        features = ['position', 'aa', 'domain', 'conservation']
    design_df = add_target_columns(sg_designs)
    feature_df_dict = dict()
    if 'position' in features:
        position_features = get_position_features(design_df)
        feature_df_dict['position'] = position_features
    if 'domain' in features:
        domain_features = get_protein_domain_features(design_df, protein_domain_df, categories=protein_domain_sources)
        feature_df_dict['domain'] = domain_features
    if 'conservation' in features:
        conservation_features = get_conservation_features(design_df, conservation_df,
                                                          conservation_column=conservation_column,
                                                          small_width=cons_small_width, large_width=cons_large_width)
        feature_df_dict['conservation'] = conservation_features
    if 'aa' in features:
        aa_features = get_amino_acid_features(design_df, aa_seq_df, width=aa_width, features=aa_features)
        feature_df_dict['aa'] = aa_features
    feature_df = design_df[['sgRNA Context Sequence', 'Transcript Base', 'AA Index']]
    for key, df in feature_df_dict.items():
        feature_df = pd.merge(feature_df, df, how='left')
    full_feature_list = list(feature_df.columns)
    remove_cols = ['sgRNA Context Sequence', 'Transcript Base', 'AA Index', 'AA Subsequence']
    for col in remove_cols:
        if col in full_feature_list:
            full_feature_list.remove(col)
    return feature_df, full_feature_list
