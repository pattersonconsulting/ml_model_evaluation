import numpy as np

import pandas as pd 
from pandas import Series


import os
import sys

from sklearn.metrics import confusion_matrix


def standard_confusion_matrix(y_true, y_pred):
	'''
		Reformat confusion matrix output from sklearn for plotting profit curve.
	'''
	[[tn, fp], [fn, tp]] = confusion_matrix(y_true, y_pred, labels=[0,1])
	return np.array([[tp, fp], [fn, tn]])






def standard_confusion_matrix_for_top_ranked_percent(y_test_list, y_probability_list, threshold, percent_ranked_instances ):
	'''
		This function is meant to test models that would be used to make N predictions (limited) rather than the whole test set;

		- this is useful when we want to see how many FPs might exist in the first N ranked predictions

		- mainly considered a helper function when evaluating how a model might be used in practice

	'''

	y_probability = np.asarray(y_probability_list)
	y_test_labels = np.array(y_test_list)



	record_count = float( len( y_test_labels ) )

	slice_index = int(percent_ranked_instances * (record_count))
	if (slice_index < 0):
		slice_index = 0

	# 1. sort both arrays by probability of predictions (highest to low)
	#thresholds = sorted(y_proba, reverse=True)

	sorted_indexs = np.argsort(-y_probability)

	sorted_y_probabilities = y_probability[sorted_indexs]
	sorted_y_test_labels = y_test_labels[sorted_indexs]



	#print( "sorted_y_probabilities: " + str(sorted_y_probabilities))
	#print( "sorted_y_test_labels  : " + str(sorted_y_test_labels))
	
	# 2. Convert predicted_prob to classification labels

	y_sorted_predicted_labels = (sorted_y_probabilities >= threshold).astype(int) #  + 0.0001

	# 3. slice arrays down to top predictions

	y_sorted_predicted_labels_sliced = y_sorted_predicted_labels[0:slice_index]
	y_sorted_test_labels_sliced = sorted_y_test_labels[0:slice_index]

	# sort thresholds such that the thresholds closest to 1.0 start towards the front of the list

	#threshold = thresholds[ threshold_index ]

	'''
	print("threshold_index: " + str(threshold_index))
	print( "thresholds: " + str(thresholds) )
	print( "threshold: " + str(threshold) )
	print( "y_proba: " + str(y_proba))
	'''

	
	
	#print( "Predictions: " + str(y_sorted_predicted_labels_sliced))
	#print( "Labels     : " + str(y_sorted_test_labels_sliced))
	

	confusion_matrix = standard_confusion_matrix(y_sorted_test_labels_sliced, y_sorted_predicted_labels_sliced)

	return confusion_matrix




def standard_confusion_matrix_for_n_ranked_instances(y_test_list, y_probability_list, threshold, top_ranked_instances_count ):
	'''
		This function is meant to test models that would be used to make N predictions (limited) rather than the whole test set;

		- this is useful when we want to see how many FPs might exist in the first N ranked predictions

		- mainly considered a helper function when evaluating how a model might be used in practice

	'''

	y_probability = np.asarray(y_probability_list)
	y_test_labels = np.array(y_test_list)



	record_count = float( len( y_test_labels ) )

	slice_index = top_ranked_instances_count
	if (slice_index < 0):
		slice_index = 0

	# 1. sort both arrays by probability of predictions (highest to low)
	#thresholds = sorted(y_proba, reverse=True)

	sorted_indexs = np.argsort(-y_probability)

	sorted_y_probabilities = y_probability[sorted_indexs]
	sorted_y_test_labels = y_test_labels[sorted_indexs]



	#print( "sorted_y_probabilities: " + str(sorted_y_probabilities))
	#print( "sorted_y_test_labels  : " + str(sorted_y_test_labels))
	
	# 2. Convert predicted_prob to classification labels

	y_sorted_predicted_labels = (sorted_y_probabilities >= threshold).astype(int) #  + 0.0001

	# 3. slice arrays down to top predictions

	y_sorted_predicted_labels_sliced = y_sorted_predicted_labels[0:slice_index]
	y_sorted_test_labels_sliced = sorted_y_test_labels[0:slice_index]

	# sort thresholds such that the thresholds closest to 1.0 start towards the front of the list

	#threshold = thresholds[ threshold_index ]

	'''
	print("threshold_index: " + str(threshold_index))
	print( "thresholds: " + str(thresholds) )
	print( "threshold: " + str(threshold) )
	print( "y_proba: " + str(y_proba))
	'''

	
	
	#print( "Predictions: " + str(y_sorted_predicted_labels_sliced))
	print( "Labels     : " + str(y_sorted_test_labels_sliced))
	

	confusion_matrix = standard_confusion_matrix(y_sorted_test_labels_sliced, y_sorted_predicted_labels_sliced)

	return confusion_matrix


def order_prediction_prob_for_n_ranked_instances(y_test_list, y_probability_list, threshold, top_ranked_instances_count ):
	'''
		This function is meant to test models that would be used to make N predictions (limited) rather than the whole test set;

		- this is useful when we want to see how many FPs might exist in the first N ranked predictions

		- mainly considered a helper function when evaluating how a model might be used in practice

	'''

	y_probability = np.asarray(y_probability_list)
	y_test_labels = np.array(y_test_list)



	record_count = float( len( y_test_labels ) )

	slice_index = top_ranked_instances_count
	if (slice_index < 0):
		slice_index = 0

	# 1. sort both arrays by probability of predictions (highest to low)
	#thresholds = sorted(y_proba, reverse=True)

	sorted_indexs = np.argsort(-y_probability)

	sorted_y_probabilities = y_probability[sorted_indexs]
	sorted_y_test_labels = y_test_labels[sorted_indexs]



	#print( "sorted_y_probabilities: " + str(sorted_y_probabilities))
	#print( "sorted_y_test_labels  : " + str(sorted_y_test_labels))
	
	# 2. Convert predicted_prob to classification labels

	y_sorted_predicted_labels = (sorted_y_probabilities >= threshold).astype(int) #  + 0.0001

	# 3. slice arrays down to top predictions

	sorted_y_probabilities_sliced = sorted_y_probabilities[0:slice_index]
	y_sorted_predicted_labels_sliced = y_sorted_predicted_labels[0:slice_index]
	y_sorted_test_labels_sliced = sorted_y_test_labels[0:slice_index]

	# sort thresholds such that the thresholds closest to 1.0 start towards the front of the list

	#threshold = thresholds[ threshold_index ]

	'''
	print("threshold_index: " + str(threshold_index))
	print( "thresholds: " + str(thresholds) )
	print( "threshold: " + str(threshold) )
	print( "y_proba: " + str(y_proba))
	'''

	
	
	#print( "Predictions: " + str(y_sorted_predicted_labels_sliced))
	#print( "Labels     : " + str(y_sorted_test_labels_sliced))
	

	#confusion_matrix = standard_confusion_matrix(y_sorted_test_labels_sliced, y_sorted_predicted_labels_sliced)

	return y_sorted_test_labels_sliced, y_sorted_predicted_labels_sliced, sorted_y_probabilities_sliced




def calc_confusion_matrix_conditional_probabilities(standard_cmatrix):
	'''
		Takes a [2, 2] np.array as input with the format: [[tp, fp], [fn, tn]]
		

		# this function calculates the
		# conditional probabilities of each cell in our confusion matrix 
		# given the probability of in our population

	'''

	[[tp, fp], [fn, tn]] = standard_cmatrix

	tp_rate = tp / (tp + fn)
	fp_rate = fp / (fp + tn)
	fn_rate = fn / (fn + tp)
	tn_rate = tn / (tn + fp)

	return np.array([[tp_rate, fp_rate], [fn_rate, tn_rate]])



def calc_confusion_matrix_estimated_probabilities(standard_cmatrix):
	'''
		Takes a [2, 2] np.array as input with the format: [[tp, fp], [fn, tn]]
		

		# this function calculates the
		# ESTIMATED probabilities of each cell in our confusion matrix 
		# given the TOTAL RECORDS in our population

	'''

	total_records = calc_total_records(standard_cmatrix)

	[[tp, fp], [fn, tn]] = standard_cmatrix

	tp_est_prob = tp / total_records
	fp_est_prob = fp / total_records
	fn_est_prob = fn / total_records
	tn_est_prob = tn / total_records

	return np.array([[tp_est_prob, fp_est_prob], [fn_est_prob, tn_est_prob]])


def calc_total_records(standard_cmatrix):

	[[tp, fp], [fn, tn]] = standard_cmatrix

	return tp + fp + fn + tn


def calc_class_priors(standard_cmatrix):

	[[tp, fp], [fn, tn]] = standard_cmatrix

	total_instances = tp + fp + fn + tn

	positive_class_prior = (tp + fn) / total_instances

	negative_class_prior = (fp + tn) / total_instances

	return positive_class_prior, negative_class_prior


'''
	Calculating Expected Profit (binary classifier)

		* this is the naive version that doesnt not take into account class priors
		* here we use "estimates of probabilities": p(h,a) == count(h,a) / TotalInstances (NOT the TP Rate, etc)

	References

		Provost, F., & Fawcett, T. (2013). Data science for business: [what you need to know about data mining and data-analytic thinking]. Sebastopol, Calif.: O'Reilly.


'''
def expected_value_calculation_naive(standard_cmatrix, cost_benefit_matrix):


	cm_est_prob = calc_confusion_matrix_estimated_probabilities( standard_cmatrix )

	[[tp_est_prob, fp_est_prob], [fn_est_prob, tn_est_prob]] = cm_est_prob


	[[tp_value, fp_value], [fn_value, tn_value]] = cost_benefit_matrix

	#print("\ntpr: " + str(tpr))
	#print("tp_value: " + str(tp_value))

	#print("fpr: " + str(fpr))
	#print("fp_value: " + str(fp_value))

	# calculate "naive" expected value (expected value without the class priors)
	expected_value = (tp_est_prob * tp_value + fn_est_prob * fn_value) + (tn_est_prob * tn_value + fp_est_prob * fp_value)


	return expected_value



'''
	Calculating Expected Value (binary classifier) with class priors

		* the p(Y|p) correspondes directly to the *true positive rate* (from confusion matrix) -- as do the other associated values
		* this is foundationally different than the "naive"-version where we use "estimates of probabilities": p(h,a) == count(h,a) / TotalInstances

	References

		Provost, F., & Fawcett, T. (2013). Data science for business: [what you need to know about data mining and data-analytic thinking]. Sebastopol, Calif.: O'Reilly.

'''
def expected_value_calculation_with_class_priors(standard_cmatrix, cost_benefit_matrix):

	

	#std_cmatrix = standard_confusion_matrix(y_test, y_pred)
	
	[[tp, fp], [fn, tn]] = standard_cmatrix

	#total_instance_count = calc_total_records( standard_cmatrix )

	cm_cond_prob = calc_confusion_matrix_conditional_probabilities( standard_cmatrix )

	[[tpr, fpr], [fnr, tnr]] = cm_cond_prob

	pos_class_prior, neg_class_prior = calc_class_priors( standard_cmatrix )

	[[tp_value, fp_value], [fn_value, tn_value]] = cost_benefit_matrix

	# formula == class_prior(positive) * [tpr * tp_value + fnr * fn_value]
	#				+ class_prior(negative) * [tnr * tn_value + fpr * fp_value]

	expected_value = ( pos_class_prior * (tpr * tp_value + fnr * fn_value) ) \
		+ ( neg_class_prior * (tnr * tn_value + fpr * fp_value) )


	return expected_value

def expected_value_calculation_with_class_priors_at_threshold( costbenefit_mat, y_proba, y_test, percent_ranked_instances ):

	record_count = float( len( y_test ) )

	threshold_index = int(percent_ranked_instances * (record_count)) - 1
	if (threshold_index < 0):
		threshold_index = 0

	'''
	print("\npercent_ranked_instances: " + str(percent_ranked_instances))
	print("threshold index: " + str(threshold_index))
	'''

	#thresholds = np.arange(0,1,0.01)

	confusion_matrices = []

	# sort thresholds such that the thresholds closest to 1.0 start towards the front of the list
	thresholds = sorted(y_proba, reverse=True)

	threshold = thresholds[ threshold_index ]

	'''
	print( "thresholds: " + str(thresholds) )
	print( "threshold: " + str(threshold) )
	print( "y_proba: " + str(y_proba))
	'''


	#print("y_proba: " + str(type(y_proba)))
	

	y_predict = (y_proba > threshold - 0.0001).astype(int) #  + 0.0001

	'''
	print( "y_predict: " + str(y_predict))
	print( "y_test   : " + str(np.array(y_test)))
	'''

	confusion_matrix = standard_confusion_matrix(y_test, y_predict)

	#print(confusion_matrix)

	expected_value_for_threshold = expected_value_calculation_with_class_priors(confusion_matrix, costbenefit_mat)

	return expected_value_for_threshold



'''
Analyze a model's performance and threshold optimal point based on a cost-benefit matrix

INPUTS:
	- cost benefit matrix in the same format as the confusion matrix above
		- format: [[tp, fp], [fn, tn]]
	- predicted probabilities
	- actual labels

OUTPUT
	{ max_profit, best_confusion_matrix }


NOTES
	we take the predicted est probabilities from a classifier and we sort them high (1.0) to low (0.0)

'''
def calculate_optimal_model_threshold( costbenefit_mat, y_proba, y_test ):
   
	profits = [] # one profit value for each threshold
	confusion_matrices = []
	thresholds = sorted(y_proba, reverse=True)

	# For each threshold, calculate profit - starting with largest threshold
	for thresh in thresholds:
		
		y_pred = (y_proba > thresh).astype(int)

		#cmatrix = confusion_matrix(y_test, y_pred)
		cmatrix = standard_confusion_matrix(y_test, y_pred)
		
		confusion_matrices.append( cmatrix )

		#tn, fp, fn, tp = cmatrix.ravel()
		
		#confusion_mat = np.array([[tp, fp], [fn, tn]])

		# Calculate total profit for this threshold
		# note: the multiplication below is a element-wise multiplication, not a matrix-multiplication (this got me one time)
		# profit = sum(sum(cmatrix * costbenefit_mat)) / len(y_test)
		
		profit = expected_value_calculation_with_class_priors( cmatrix, costbenefit_mat )
		profits.append( profit )

	
	max_profit = max( profits )

	max_profit_index = profits.index( max( profits ) )

	#max_profit_threshold = thresholds[ max_profit_index ]

	#print("total profit entries: " + str(len(profits)))
	#print("total confusion_matrices entries: " + str(len(confusion_matrices)))

	cfmtx_max = confusion_matrices[ max_profit_index ]

	return max_profit, cfmtx_max, max_profit_index


def calculate_max_profit_record_percent_for_model( costbenefit_mat, y_proba, y_test ):


	xaxis_percents = np.linspace(0, 1.0, len(y_proba))# len(profit_data))

	

	max_model_expected_value_tracker = []

	for percent in xaxis_percents:

		expected_value_for_threshold = expected_value_calculation_with_class_priors_at_threshold( costbenefit_mat, y_proba, y_test, percent )

		max_model_expected_value_tracker.append( { 'ev': expected_value_for_threshold, 'percent': percent } )


	model_max_point = max(max_model_expected_value_tracker, key=lambda x:x['ev'])

	print("Most valuable ev point: ")
	print(model_max_point["ev"])
	print(model_max_point["percent"])

	return model_max_point["ev"], model_max_point["percent"]


def calculate_optimal_model_threshold_for_budget( costbenefit_mat, y_proba, y_test, total_budget ):

	# TODO: implement

	return 0


