import pandas as pd


class ModelPrediction():
    def __init__(self, prediction_file, prediction_col, name, train_sgrna_file, train_20mer_col=None,
                 train_23mer_col=None, train_30mer_col=None, classtype='model', target_col=None,
                 merge_sgrna_file=False, model_30mer_col=None):
        self.prediction_file = prediction_file
        self.prediction_col = prediction_col
        self.name = name
        self.train_sgrna_file = train_sgrna_file
        self.train_20mer_col = train_20mer_col
        self.train_23mer_col = train_23mer_col
        self.train_30mer_col = train_30mer_col
        self.classtype = classtype
        self.target_col = target_col
        self.prediction_df = None
        self.sgrnas = None
        self.merge_sgrna_file = merge_sgrna_file
        self.model_30mer_col = model_30mer_col

    def load_model_predictions(self):
        if not self.merge_sgrna_file:
            self.prediction_df = pd.read_csv(self.prediction_file)
        else:
            prediction_df = pd.read_csv(self.prediction_file)
            sgrna_df = pd.read_csv('../data/interim/rs_dev_all_sgrnas.csv')
            merged_predictions = (prediction_df
                                  .merge(sgrna_df, how='inner',
                                         left_on=self.model_30mer_col, right_on='sgRNA Context Sequence'))
            out_predictions = merged_predictions[['sgRNA Sequence', 'sgRNA Context Sequence', 'PAM Sequence',
                                                  self.prediction_col]].copy()
            self.prediction_df = out_predictions

    def set_sgrnas(self):
        train_df = pd.read_csv(self.train_sgrna_file)
        if self.train_20mer_col is not None:
            self.sgrnas = set(train_df[self.train_20mer_col].to_list())
        if self.train_23mer_col is not None:
            self.sgrnas = set(train_df[self.train_23mer_col]
                              .str[:20]
                              .to_list())
        if self.train_30mer_col is not None:
            self.sgrnas = set(train_df[self.train_30mer_col]
                              .str[4:-6]
                              .to_list())


rule_set2_predictions = ModelPrediction(prediction_file='../data/external/Rule_Set_2_rs_dev_all_sgrnas.csv',
                                        prediction_col='Rule Set 2', name='Rule Set 2',
                                        train_sgrna_file='https://raw.githubusercontent.com/MicrosoftResearch/Azimuth/master/azimuth/data/FC_plus_RES_withPredictions.csv',
                                        train_30mer_col='30mer')

deepspcas9_predictions = ModelPrediction(prediction_file='../data/external/DeepSpCas9_rs_dev_all_sgrnas.csv',
                                         prediction_col='DeepSpCas9', name='DeepSpCas9',
                                         train_sgrna_file='../data/raw/Kim2019_S1_Train.csv',
                                         train_30mer_col='Target context sequence (4+20+3+3)')

deepcrispr_predictions = ModelPrediction(prediction_file='../data/external/DeepCRISPR_rs_dev_all_sgrnas.csv',
                                         prediction_col='DeepCRISPR', name='DeepCRISPR',
                                         train_sgrna_file='../data/external/Chuai2018_tableS5_sgrnas.csv',
                                         train_23mer_col='sgRNA')

vbc_activity_predictions = ModelPrediction(prediction_file='../data/external/vbc_activity_scores_rs_dev.csv',
                                           prediction_col='VBC Activity', name='VBC Activity',
                                           train_sgrna_file='../data/external/munozS4_azimuth_sgrnas.csv',
                                           train_20mer_col='sgRNA Sequence')

crispron_activity_predictions = ModelPrediction(prediction_file='../data/external/crispron_rs_dev_all_sgrnas.csv',
                                                prediction_col='CRISPRon', name='CRISPRon',
                                                train_sgrna_file='../data/external/CRISPRon_train.csv',
                                                train_20mer_col='sgRNA Sequence',
                                                merge_sgrna_file=True, model_30mer_col='30mer')

model_prediction_list = [crispron_activity_predictions, 
                         deepspcas9_predictions, rule_set2_predictions,
                         vbc_activity_predictions, deepcrispr_predictions]

vbc_predictions = ModelPrediction('../data/external/vbc_scores_rs_dev.csv', 'VBC score', 'VBC score',
                                  '../data/external/munozS4_azimuth_sgrnas.csv', train_20mer_col='sgRNA Sequence',
                                  target_col='gene')
