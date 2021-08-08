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

def get_position_features(sg_df, id_cols):
    """Get  features ['dist from start', 'dist from end', 'dist percent', 'sense']

    :param sg_df: DataFrame
    :param id_cols: list
    :return: DataFrame
    """
    position_df = sg_df[id_cols + ['Target Cut %']].copy()
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
                    'Isoelectric Point', 'Secondary Structure']
    seq_len = len(aa_sequences[0])
    sequence_order = [str(int(x - seq_len//2)) for x in range(seq_len)]
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


def extract_amino_acid_subsequence(sg_aas, width):
    """ Get the amino acid subsequence with a width of `width` on either side of the Amino Acid index

    :param sg_aas: DataFrame, sgRNA designs merged with amino acid sequence
    :param width: int
    :return: DataFrame
    """
    # Pad the sequences at the beginning and end, so our index doesn't go over
    l_padding = '-' * width
    r_padding = '-' * (width - 1)
    # add stop codon at the end of the sequence
    sg_aas_subseq = sg_aas.copy()
    sg_aas_subseq['extended_seq'] = l_padding + sg_aas_subseq['seq'] + '*' + r_padding
    sg_aas_subseq['AA 0-Indexed'] = sg_aas_subseq['AA Index'] - 1
    sg_aas_subseq['AA 0-Indexed padded'] = sg_aas_subseq['AA 0-Indexed'] + width
    sg_aas_subseq['seq_start'] = sg_aas_subseq['AA 0-Indexed padded'] - width
    sg_aas_subseq['seq_end'] = sg_aas_subseq['AA 0-Indexed padded'] + width
    sg_aas_subseq['AA Subsequence'] = sg_aas_subseq.apply(lambda row: row['extended_seq'][row['seq_start']:(row['seq_end'] + 1)],
                                                    axis=1)
    return sg_aas_subseq



def get_amino_acid_features(sg_designs, aa_seq_df, width, features, id_cols):
    """Featurize amino acid sequences

    :param sg_designs: DataFrame
    :param aa_seq_df: DataFrame, Transcript Base and (AA) seq
    :param width: int, length on each side of the cut site
    :param feature: list
    :return: DataFrame
    """
    sg_aas = (aa_seq_df.merge(sg_designs[id_cols + ['Transcript Base', 'AA Index']],
                              how='inner',
                              on=['Transcript Base', 'Target Transcript']))
    sg_aas_subseq = extract_amino_acid_subsequence(sg_aas, width)
    aa_features = featurize_aa_seqs(sg_aas_subseq['AA Subsequence'], features=features)
    aa_features_annot = pd.concat([sg_aas_subseq[id_cols + ['AA Subsequence']]
                                   .reset_index(drop=True),
                                   aa_features.reset_index(drop=True)], axis=1)
    return aa_features_annot


# Protein Domain

def get_protein_domain_features(sg_design_df, protein_domains, sources, id_cols):
    """Get binary dataframe of protein domains

    :param sg_design_df: DataFrame, with columns ['Transcript Base', 'sgRNA Context Sequence', 'AA Index']
    :param protein_domains: DataFrame, with columns ['Transcript Base', 'type']
    :param sources: list. list of database types to include
    :param id_cols: list
    :return: DataFrame, with binary features for protein domains
    """
    if sources is None:
        sources = ['Pfam', 'PANTHER', 'HAMAP', 'SuperFamily', 'TIGRfam', 'ncoils', 'Gene3D',
                   'Prosite_patterns', 'Seg', 'SignalP', 'TMHMM', 'MobiDBLite',
                   'PIRSF', 'PRINTS', 'Smart', 'Prosite_profiles']  # exclude sifts
    protein_domains = protein_domains[protein_domains['type'].isin(sources)]
    clean_designs = sg_design_df[id_cols + ['Transcript Base', 'AA Index']].copy()
    designs_domains = clean_designs.merge(protein_domains,
                                          how='inner', on='Transcript Base')
    # Note - not every sgRNA will be present in the feature df
    filtered_domains = (designs_domains[designs_domains['AA Index'].between(designs_domains['start'],
                                                                            designs_domains['end'])]
                        .copy())
    filtered_domains = filtered_domains[id_cols + ['type']].drop_duplicates()
    filtered_domains['present'] = 1
    domain_feature_df = (filtered_domains.pivot_table(values='present',
                                                      index=id_cols,
                                                      columns='type', fill_value=0)
                         .reset_index())
    # Ensure all domain columns are present for testing
    full_column_df = pd.DataFrame(columns=id_cols + sources, dtype=int)  # empty
    domain_feature_df = pd.concat([full_column_df, domain_feature_df])
    return domain_feature_df


# Conservation

def get_conservation_ranges(cut_pos, small_width, large_width):
    small_range = range(cut_pos - small_width, cut_pos + small_width)
    large_range = range(cut_pos - large_width, cut_pos + large_width)
    return small_range, large_range


def get_conservation_features(sg_designs, conservation_df, conservation_column,
                              small_width, large_width, id_cols):
    """Get conservation features

    :param sg_designs: DataFrame
    :param conservation_df: DataFrame, tidy conservation scores indexed by Transcript Base and target position
    :param conservation_column: str, name of column to calculate scores with
    :param small_width: int, small window length to average scores in one direction
    :param large_width: int, large window length to average scores in the one direction
    :return: DataFrame of conservation features
    """
    sg_designs_width = sg_designs[id_cols + ['Transcript Base']].copy()
    sg_designs_width['target position small'], sg_designs_width['target position large'] =  \
        zip(*sg_designs_width['Target Cut Length']
            .apply(get_conservation_ranges, small_width=small_width,
                   large_width=large_width))
    small_width_conservation = (sg_designs_width.drop('target position large', axis=1)
                                .rename({'target position small': 'target position'}, axis=1)
                                .explode('target position')
                                .merge(conservation_df, how='inner',
                                       on=['Target Transcript', 'Transcript Base', 'target position'])
                                .groupby(id_cols)
                                .agg(cons=(conservation_column, 'mean'))
                                .rename({'cons': 'cons_' + str(small_width * 2)}, axis=1)
                                .reset_index())
    large_width_conservation = (sg_designs_width.drop('target position small', axis=1)
                                .rename({'target position large': 'target position'}, axis=1)
                                .explode('target position')
                                .merge(conservation_df, how='inner',
                                       on=['Target Transcript', 'Transcript Base', 'target position'])
                                .groupby(id_cols)
                                .agg(cons=(conservation_column, 'mean'))
                                .rename({'cons': 'cons_' + str(large_width * 2)}, axis=1)
                                .reset_index())
    cons_feature_df = small_width_conservation.merge(large_width_conservation, how='outer',
                                                     on=id_cols)
    return cons_feature_df


# Build Target Features

def build_target_feature_df(sg_designs, features=None,
                            aa_seq_df=None, aa_width=16, aa_features=None,
                            protein_domain_df=None, protein_domain_sources=None,
                            conservation_df=None, conservation_column='ranked_conservation',
                            cons_small_width=4, cons_large_width=32,
                            id_cols=None):
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
    if id_cols is None:
        id_cols = ['sgRNA Context Sequence', 'Target Cut Length',
                   'Target Transcript', 'Orientation', 'dataset']
    design_df = add_target_columns(sg_designs)
    feature_df_dict = dict()
    if 'position' in features:
        position_features = get_position_features(design_df, id_cols)
        feature_df_dict['position'] = position_features
    if 'domain' in features:
        domain_features = get_protein_domain_features(design_df, protein_domain_df, protein_domain_sources, id_cols)
        feature_df_dict['domain'] = domain_features
    if 'conservation' in features:
        conservation_features = get_conservation_features(design_df, conservation_df,
                                                          conservation_column,
                                                          cons_small_width, cons_large_width,
                                                          id_cols)
        feature_df_dict['conservation'] = conservation_features
    if 'aa' in features:
        aa_features = get_amino_acid_features(design_df, aa_seq_df, aa_width, aa_features, id_cols)
        feature_df_dict['aa'] = aa_features
    feature_df = design_df[id_cols]
    for key, df in feature_df_dict.items():
        feature_df = pd.merge(feature_df, df, how='left', on=id_cols)
    full_feature_list = list(feature_df.columns)
    remove_cols = id_cols + ['Transcript Base', 'AA Index', 'AA Subsequence']
    for col in remove_cols:
        if col in full_feature_list:
            full_feature_list.remove(col)
    return feature_df, full_feature_list
