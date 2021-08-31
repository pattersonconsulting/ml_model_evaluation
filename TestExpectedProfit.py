import unittest

import numpy as np
from sklearn.metrics import confusion_matrix

import model_valuation


class TestExpectedProfit(unittest.TestCase):

    def test_sum(self):
        self.assertEqual(sum([1, 2, 3]), 6, "Should be 6")


    def test_standard_confusion_matrix(self):

        y_true = [0, 1, 0, 1]
        y_pred = [1, 1, 1, 0]

        scmtrx = model_valuation.standard_confusion_matrix(y_true, y_pred)

        #conf_mat = confusion_matrix( y_true, y_pred )

        [[tp, fp], [fn, tn]] = scmtrx


        
        print("TP: " + str(tp))
        print("TN: " + str(tn))
        print("FP: " + str(fp))
        print("FN: " + str(fn))



        self.assertEqual(tp, 1, "TP Should be 1")
        self.assertEqual(tn, 0, "TN Should be 0")

        self.assertEqual(fp, 2, "FP Should be 2")


    def test_standard_confusion_matrix_cond_prob(self):

        y_true = [0, 1, 0, 1]
        y_pred = [1, 1, 1, 0]

        scmtrx = model_valuation.standard_confusion_matrix(y_true, y_pred)

        [[tpr, fpr], [fnr, tnr]] = model_valuation.calc_confusion_matrix_conditional_probabilities(scmtrx)


        self.assertEqual(tpr, 0.5, "TPR Should be 0.5")
        self.assertEqual(tnr, 0, "TNR Should be 0")

        self.assertEqual(fpr, 1, "FPR Should be 1")

        print("Testing confusion matrix conditional probabilities")



    def test_standard_confusion_matrix_class_priors(self):

        print("Testing confusion matrix class priors")

        y_true = [0, 1, 0, 1]
        y_pred = [1, 1, 1, 0]

        scmtrx = model_valuation.standard_confusion_matrix(y_true, y_pred)

        pos_class_prior, neg_class_prior = model_valuation.calc_class_priors(scmtrx)



        self.assertEqual(pos_class_prior, 0.5, "positive class prior Should be 0.5")
        self.assertEqual(neg_class_prior, 0.5, "negative class prior Should be 0.5")

    '''
    def test_expected_profit_naive(self):

        y_true = [0, 1, 0, 1]
        y_pred = [1, 1, 1, 0]

        conf_mat = confusion_matrix( y_true, y_pred )

        # confusion_mat = np.array([[tp, fp], [fn, tn]])
        cost_benefit_matrix = np.array([[4, -5],
                            [0, 0]])

        tn, fp, fn, tp = conf_mat.ravel()
        print("TP: " + str(tp))
        print("TN: " + str(tn))
        print("FP: " + str(fp))
        print("FN: " + str(fn))


        #expected_profit = model_valuation.expected_profit_calc(y_true, y_pred, cost_benefit_matrix)


        self.assertEqual(tp, 1, "TP Should be 1")
    '''

    def test_expected_value_naive(self):

        print("\nTesting expected value naive")


        y_true = [0, 1, 0, 1]
        y_pred = [1, 1, 1, 0]

        scmtrx = model_valuation.standard_confusion_matrix(y_true, y_pred)


        # confusion_mat = np.array([[tp, fp], [fn, tn]])
        cost_benefit_matrix = np.array([[4, -5],
                            [0, 0]])


        exp_value = model_valuation.expected_value_calculation_naive(scmtrx, cost_benefit_matrix)

        print("naive expected value: " + str(exp_value))

        self.assertEqual(exp_value, -1.5, "Expected Value should be -3.0")



    def test_expected_value_w_priors(self):

        print("\nTesting expected value with class priors")


        y_true = [0, 1, 0, 1]
        y_pred = [1, 1, 1, 0]

        scmtrx = model_valuation.standard_confusion_matrix(y_true, y_pred)


        # confusion_mat = np.array([[tp, fp], [fn, tn]])
        cost_benefit_matrix = np.array([[4, -5],
                            [0, 0]])


        exp_value = model_valuation.expected_value_calculation_with_class_priors(scmtrx, cost_benefit_matrix)

        print("expected value: " + str(exp_value))


        self.assertEqual(exp_value, -1.5, "Expected Value should be -1.5")


    def test_expected_value_w_priors_2(self):

        print("Testing expected value with class priors alt set up")

        #[[tp, fp], [fn, tn]] = standard_cmatrix

        std_cmatrix = [[15, 4], [40, 1441]]


        # confusion_mat = np.array([[tp, fp], [fn, tn]])
        cost_benefit_matrix = np.array([[2440, -60],
                                        [0, 0]])


        exp_value = model_valuation.expected_value_calculation_with_class_priors(std_cmatrix, cost_benefit_matrix)

        print("imbalanced dataset results, expected value: " + str(exp_value))


        self.assertEqual(exp_value, 24.24, "Expected Value should be 24.24")



    def test_expected_value_w_priors_and_percent_instances(self):

        print("Testing expected value with class priors set up and threshold param")


        y_true = [1, 1, 1, 1, 1, 1, 0, 0, 0, 1]
        y_pred = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0])

        #scmtrx = model_valuation.standard_confusion_matrix(y_true, y_pred)


        # confusion_mat = np.array([[tp, fp], [fn, tn]])
        cost_benefit_matrix = np.array([[4, -5],
                            [0, 0]])

        exp_value = model_valuation.expected_value_calculation_with_class_priors_at_threshold( cost_benefit_matrix, y_pred, y_true, 1.0 )
        self.assertEqual(np.round(float(exp_value),2), 1.3, "Expected Value should be 1.3")
        #self.assertEqual(exp_value, 1.29, "Expected Value should be 1.299")

        #exp_value2 = model_valuation.expected_value_calculation_with_class_priors_at_threshold( cost_benefit_matrix, y_pred, y_true, 0.0 )
        #self.assertEqual(exp_value2, 0.0, "Expected Value should be 0.0")

        exp_value2 = model_valuation.expected_value_calculation_with_class_priors_at_threshold( cost_benefit_matrix, y_pred, y_true, 0.7 )
        
        self.assertEqual(exp_value2, 1.9, "Expected Value should be 1.9")


        exp_value3 = model_valuation.expected_value_calculation_with_class_priors_at_threshold( cost_benefit_matrix, y_pred, y_true, 0.3 )
        
        self.assertEqual(exp_value3, 1.2, "Expected Value should be 1.2")        



    def test_find_max_value_record_percent(self):


        y_true = [1, 1, 1, 1, 1, 1, 0, 0, 0, 1]
        y_pred = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0])

        # confusion_mat = np.array([[tp, fp], [fn, tn]])
        cost_benefit_matrix = np.array([[4, -5],
                            [0, 0]])

        expected_value, percent_data = model_valuation.calculate_max_profit_record_percent_for_model( cost_benefit_matrix, y_pred, y_true )        

        print("Testing finding max value and record percent:")
        print("Expected Value: " + str(expected_value))
        print("At Percent of Data: " + str(percent_data))


if __name__ == '__main__':
    unittest.main()