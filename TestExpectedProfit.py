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


    def test_expected_value_w_priors(self):

        print("Testing expected value with class priors")


        y_true = [0, 1, 0, 1]
        y_pred = [1, 1, 1, 0]

        scmtrx = model_valuation.standard_confusion_matrix(y_true, y_pred)


        # confusion_mat = np.array([[tp, fp], [fn, tn]])
        cost_benefit_matrix = np.array([[4, -5],
                            [0, 0]])


        exp_value = model_valuation.expected_value_calculation_with_class_priors(scmtrx, cost_benefit_matrix)

        print("expected value: " + str(exp_value))





if __name__ == '__main__':
    unittest.main()