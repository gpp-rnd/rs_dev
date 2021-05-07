from scipy import stats
import pandas as pd
import plotnine as gg


def get_predictive_performance(prediction_list, activity_col='sgRNA Activity'):
    """From a list of dataframes of predictions for each fold for each datasetset for each guide,
    get pearson correlation of each fold/dataset

    :param prediction_list: list of DataFrame
    :return: DataFrame
    """
    test_prediction_df = (pd.concat(prediction_list)
                          .rename({'dataset': 'testing_set'}, axis=1))
    predictive_performance = (test_prediction_df.groupby(['model_name', 'testing_set', 'fold'])
                              .apply(lambda df: stats.pearsonr(df[activity_col], df['prediction'])[0])
                              .reset_index(name='pearson_r'))
    predictive_performance['relative_performance'] = (predictive_performance.groupby(['fold', 'testing_set'])
                                                      ['pearson_r']
                                                      .transform(lambda x: x/x.max()))
    median_relative_performance = (predictive_performance.groupby(['model_name'])
                                   .agg(median_performance = ('relative_performance', 'median'))
                                   .reset_index()
                                   .sort_values('median_performance'))
    predictive_performance['model_name'] = pd.Categorical(predictive_performance['model_name'],
                                                            categories=median_relative_performance['model_name'])
    if predictive_performance['testing_set'].isin(predictive_performance['model_name']).all():
        predictive_performance['testing_set'] = pd.Categorical(predictive_performance['testing_set'],
                                                               categories=median_relative_performance['model_name'])
    predictive_performance['fold'] = predictive_performance['fold'].astype('category')
    return predictive_performance


def plot_pearson_heatmap(predictive_performance):
    """Plot heatmap of pearson correlations

    :param predictive_performance: DataFrame from `get_predictive_performance`
    :return: plotnine figure
    """
    g = (gg.ggplot(predictive_performance) +
         gg.aes(x='testing_set', y='model_name', fill='pearson_r') +
         gg.geom_tile(color='black') +
         gg.scale_fill_cmap('RdBu_r', limits=(-1, 1)) +
         gg.theme_classic() +
         gg.theme(axis_text_x=gg.element_text(angle=90, hjust=0.5, vjust=1)) +
         gg.facet_wrap('fold'))
    return g


def plot_relative_performance(predictive_performance):
    """Plot boxplot of relative performance

    :param predictive_performance: predictive_performance: DataFrame from `get_predictive_performance`
    :return: plotnine figure
    """
    g = (gg.ggplot(predictive_performance) +
         gg.aes(x='model_name', y='relative_performance') +
         gg.geom_point(gg.aes(color='fold'), position=gg.position_dodge(width=0.5)) +
         gg.scale_color_brewer(type='qual', palette='Set2') +
         gg.geom_boxplot(fill=None, outlier_alpha=0) +
         gg.theme_classic() +
         gg.theme(axis_text_x=gg.element_text(angle=90, hjust=0.5, vjust=1)))
    return g


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


def featurize_aa_seqs(aa_sequences, features=None):
    if features is None:
        features = ['Pos. Ind. 1mer', 'Pos. Ind. 2mer',
                    'Pos. Dep. 1mer']
    # TODO - take center position into account
    sequence_order = list(range(len(aa_sequences[0])))
    aas = ['A', 'C', 'D', 'E', 'F',
           'G', 'H', 'I', 'K', 'L',
           'M', 'N', 'P', 'Q', 'R',
           'S', 'T', 'V', 'W', 'Y']
    feature_dict_list = []
    for i, aa_sequence in enumerate(aa_sequences):
        feature_dict = {}
        if 'Pos Ind. 1mer' in features:
            get_one_aa_frac(feature_dict, aa_sequence, aas)
        if 'Pos. Ind. 2mer' in features:
            get_two_aa_frac(feature_dict, aa_sequence, aas)
        if 'Pos. Dep. 1mer' in features:
            get_one_aa_pos(feature_dict, aa_sequence, aas, sequence_order)
        feature_dict_list.append(feature_dict)
    feature_matrix = pd.DataFrame(feature_dict_list)
    feature_matrix.index = aa_sequences
    return feature_matrix