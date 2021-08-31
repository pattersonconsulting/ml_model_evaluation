import numpy as np

import pandas as pd 
from pandas import Series


import os
import sys

from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import plot_precision_recall_curve
from sklearn.metrics import roc_curve, auc, confusion_matrix

import matplotlib.pyplot as plt

import model_valuation

import locale
locale.setlocale( locale.LC_ALL, '' )

'''
Plots ROC Curve for multiple classifiers

INPUTS:
	- tuple of model data
		- model name
		- predicted probabilities
		- actual labels

OUTPUT
	- plotted graph png

'''
def plot_roc_curves( model_data_tuple, file_name ):

	
	plt.figure(1)
	plt.plot([0, 1], [0, 1], 'k--')


	for (y_test, y_proba, model_name) in model_data_tuple:

		fpr, tpr, thresholds = roc_curve( y_test, y_proba )

		auc_score = auc(fpr, tpr)

		plt.plot(fpr, tpr, label= (model_name + ', AUC: {0:.2f}'.format(auc_score)))




	plt.xlabel('False positive rate')
	plt.ylabel('True positive rate')
	plt.title('ROC curve')
	plt.legend(loc='best')
	plt.savefig('./graphs/' + file_name + '.png')
	plt.close()

	# TODO


'''

	TODO: RECHECK the cumulative profit calcs

	Notes

		- if every model is predicting the same set of test records then the linear space of the x-axis will be consistent across graphed plots

'''
def plot_profit_curves( costbenefit_mat, model_data_tuple, file_name, **kwargs ):

	[[tp_value, fp_value], [fn_value, tn_value]] = costbenefit_mat

	total_budget = kwargs.get('total_budget', None)
	#cost_per_action = kwargs.get('cost_per_action', None) # already have this in hte cost benefit matrix
	cost_per_action = fp_value

	
	plt.figure(1)
	fig, ax = plt.subplots(1,1,figsize = (10,10))

	ax.grid(alpha = .4,color = 'grey',linestyle = ':')

	#plt.plot([0, 1], [0, 1], 'k--')

	max_model_profit_tracker = []

	total_records = 0


	for (y_test, y_proba, model_name) in model_data_tuple:

		print("Rendering profit curve for: " + model_name)

		# the profit data contains, for each threshold, the exected value ("profit") of the classifier at that threshold/confusion matrix point
		#profit_data, thresholds, confusion_matrices = calculate_profit_curve_points( costbenefit_mat, y_proba, y_test )
		profit_data, confusion_matrices = calculate_profit_curve_points_via_percents( costbenefit_mat, y_proba, y_test )

		max_profit, cfmtx_max, max_profit_index = model_valuation.calculate_optimal_model_threshold( costbenefit_mat, y_proba, y_test )

	
		# goal: plot "percentage of test instances decreasing by score"

		xaxis_linspace = np.linspace(0, 100, len(profit_data))

		total_records = len(profit_data)

		plt.plot(xaxis_linspace, profit_data, label = ('{}, max profit ' + locale.currency( max_profit, grouping=True ) + ' ').format(model_name))

		max_model_profit_tracker.append( { 'profit': max_profit, 'max_profit_index': max_profit_index, 'xaxix_linspace': xaxis_linspace } )


	ylim0, ylim1 = ax.get_ylim()
	ax.autoscale(enable=True, axis='x', tight=True)


	# shade the negative profit area with red
	ax.axhspan(0, ylim0, alpha=0.2, hatch='/', color='red')



	#ind_max_profit = np.argmax( max_model_profit_tracker )
	mostValuableModelMax = max(max_model_profit_tracker, key=lambda x:x['profit'])


	print("Most valuable model: ")
	print(mostValuableModelMax["profit"])
	print(mostValuableModelMax["max_profit_index"])

	linspace_range_xaxis = mostValuableModelMax["xaxix_linspace"]

	#ax.axvline(mostValuableModelMax["threshold"], 0, ylim1,linestyle = '--', color = 'k',label = 'Max Profit')

	#ind = np.argmax(profits)
	ax.axvline( linspace_range_xaxis[ mostValuableModelMax["max_profit_index"] ], 0, mostValuableModelMax["profit"], linestyle = '--', color = 'k',label = 'Max Profit Line')	
	


	if total_budget is not None:

		max_actions_budgeted = total_budget / (cost_per_action * -1)

		percent_of_data_budgeted = (max_actions_budgeted * total_records)


		print("total_budget: " + str(total_budget))

		print("cost_per_action: " + str(cost_per_action))

		print("max_actions_budgeted: " + str(max_actions_budgeted))

		print("percent_of_data_budgeted: " + str(percent_of_data_budgeted))

		ax.axvline( percent_of_data_budgeted, 0, mostValuableModelMax["profit"], linestyle = '--', color = 'green',label = 'Max Budget Line')	
	

	#ax.plot([-5, 110], [0, 0], color='navy', linestyle='--',lw = 2);

	#plt.xlabel('False positive rate')
	#plt.ylabel('True positive rate')
	#plt.title('ROC curve')
	#plt.legend(loc='best')

	#plt.ylim(0,8.5)
	plt.title("Profit Curve")
	plt.legend(loc='best')
	plt.xlabel("Percentage of test instances (decreasing by score)")
	plt.ylabel("Profit ($/action)")

	plt.savefig('./graphs/' + file_name + '.png')
	plt.close()


'''
Calculates the (x,y) data for a profit curve

INPUTS:
	

OUTPUT


NOTE

	- WHAT happens if the threshold is 1.0 and the prob is 1.0?

Implementation Notes
	
	References: 

		[1] "Data Science for Business", O'Reilly 2013, Provost and Fawcett

			-	Expected Value (with class priors) concept and equations: p.201
			-	Profit Curve concepts: p.212

				"Each curve is based on the idea of examining the effects of thresholding the value of a classifier at successive points, 
					implicitly dividing the list of instances into many successful sets of predictive positive and negative instances."

				"As we move the threshold 'down' the ranking, we get additional instances predicted as being positive rather than negative"

				"each threshol ... will have a corresponding confusion matrix. The expected value equation can then be used to take each of these 
					confusion matrices and generated an expected value for the successive points along the 'profit curve'"

				Other notes:

				* we rank the probabilities of the classifer's prediction on the test set in a decreasing fashion
				* we then calculate the expected profit moving the threshold point down the list of probabilities and re-calculating the confusion matrix for each successive cut-point in the list
					* this makes the x-axis the "percent of test instances (decreasing by score)"
				* at each cut point we record the expected profit and the percent of instances included in the cut point (x,y)
					* graphing these values give us a profit curve

'''
def calculate_profit_curve_points( costbenefit_mat, y_proba, y_test ):

	record_count = float( len( y_test ) )

	#thresholds = np.arange(0,1,0.01)

	confusion_matrices = []

	# sort thresholds such that the thresholds closest to 1.0 start towards the front of the list
	thresholds = sorted(y_proba, reverse=True)
	#thresholds = np.insert(thresholds,0,0)

	#fpr, tpr, thresholds = roc_curve( y_test, y_proba )

	#print("Thresholds sorted: ")
	#print( thresholds[:5] )

	profit_data = []
	#x_axis_percentages = []

	print("y_proba: " + str(type(y_proba)))

	for threshold in thresholds:

		#print(threshold)
		#y_predict = y_proba >= threshold

		y_predict = (y_proba > threshold).astype(int) #  + 0.0001

		#print( y_predict[:5] )


		confusion_matrix = model_valuation.standard_confusion_matrix(y_test, y_predict)

		confusion_matrices.append( confusion_matrix )

		expected_value_for_threshold = model_valuation.expected_value_calculation_with_class_priors(confusion_matrix, costbenefit_mat)

		profit_data.append(expected_value_for_threshold)

	return profit_data, thresholds, confusion_matrices



def calculate_profit_curve_points_via_percents( costbenefit_mat, y_proba, y_test ):


	confusion_matrices = []


	xaxis_percents = np.linspace(0, 1.0, len(y_proba))# len(profit_data))

	profit_data = []
	#x_axis_percentages = []

	#print("y_proba: " + str(type(y_proba)))

	for percent in xaxis_percents:

		expected_value_for_threshold = model_valuation.expected_value_calculation_with_class_priors_at_threshold( costbenefit_mat, y_proba, y_test, percent )

		#confusion_matrix = model_valuation.standard_confusion_matrix(y_test, y_predict)

		#confusion_matrices.append( confusion_matrix )

		profit_data.append(expected_value_for_threshold)

	return profit_data, confusion_matrices





'''
Plots PR Curve for multiple classifiers

INPUTS:
	- array of models
		- predicted probabilities
		- actual labels

OUTPUT
	- plotted graph png


References

	https://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html

'''
def plot_pr_curve( classifier, X_test, y_proba, y_test, file_name ):

	plt.figure(1)

	#y_predict = (y_proba > threshold).astype(int)

	average_precision = average_precision_score(y_test, y_proba)

	print('Average precision-recall score: {0:0.2f}'.format(average_precision))

	disp = plot_precision_recall_curve(classifier, X_test, y_test)

	disp.ax_.set_title('2-class Precision-Recall curve: AP={0:0.2f}'.format(average_precision))

	#disp.ax_.savefig('./graphs/' + file_name + '.png')
	#disp.ax_.close()

	#plt.title('ROC curve')
	plt.legend(loc='best')
	plt.savefig('./graphs/' + file_name + '.png')
	plt.close()


	# TODO
	return 0


