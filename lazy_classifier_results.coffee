  3%|▎         | 1/29 [00:41<19:08, 41.02s/it]{'Model': 'AdaBoostClassifier', 'Accuracy': 0.9991222218320986, 'Balanced Accuracy': 0.8722291568568903, 'ROC AUC': 0.8722291568568903, 'F1 Score': 0.9991222218320986, 'recall_score': 0.7448979591836735, 'Time taken': 41.023133993148804}
  7%|▋         | 2/29 [02:07<30:28, 67.73s/it]{'Model': 'BaggingClassifier', 'Accuracy': 0.9995786664794073, 'Balanced Accuracy': 0.9030172599428066, 'ROC AUC': 0.9030172599428067, 'F1 Score': 0.9995624872740402, 'recall_score': 0.8061224489795918, 'Time taken': 86.42220902442932}
 10%|█         | 3/29 [02:07<16:01, 36.99s/it]{'Model': 'BernoulliNB', 'Accuracy': 0.9990519995786665, 'Balanced Accuracy': 0.8161682582430834, 'ROC AUC': 0.8161682582430834, 'F1 Score': 0.9990041419823578, 'recall_score': 0.6326530612244898, 'Time taken': 0.3980598449707031}
 14%|█▍        | 4/29 [02:10<09:46, 23.48s/it]{'Model': 'CalibratedClassifierCV', 'Accuracy': 0.9991573329588147, 'Balanced Accuracy': 0.8162210156994706, 'ROC AUC': 0.8162210156994706, 'F1 Score': 0.9990986311301908, 'recall_score': 0.6326530612244898, 'Time taken': 2.769416332244873}
 21%|██        | 6/29 [02:26<05:53, 15.38s/it]{'Model': 'DecisionTreeClassifier', 'Accuracy': 0.9991573329588147, 'Balanced Accuracy': 0.8773399905826146, 'ROC AUC': 0.8773399905826146, 'F1 Score': 0.9991573329588147, 'recall_score': 0.7551020408163265, 'Time taken': 15.893152952194214}
 24%|██▍       | 7/29 [02:27<03:49, 10.42s/it]{'Model': 'DummyClassifier', 'Accuracy': 0.9982795547909132, 'Balanced Accuracy': 0.5, 'ROC AUC': 0.5, 'F1 Score': 0.9974200728063972, 'recall_score': 0.0, 'Time taken': 0.21435022354125977}
 28%|██▊       | 8/29 [02:27<02:31,  7.19s/it]{'Model': 'ExtraTreeClassifier', 'Accuracy': 0.9992099996488887, 'Balanced Accuracy': 0.8824596172177368, 'ROC AUC': 0.8824596172177368, 'F1 Score': 0.999207977480133, 'recall_score': 0.7653061224489796, 'Time taken': 0.2786588668823242}
 31%|███       | 9/29 [02:39<02:52,  8.62s/it]{'Model': 'ExtraTreesClassifier', 'Accuracy': 0.9996313331694814, 'Balanced Accuracy': 0.9132301344848576, 'ROC AUC': 0.9132301344848576, 'F1 Score': 0.999618259514336, 'recall_score': 0.826530612244898, 'Time taken': 11.766731977462769}
 34%|███▍      | 10/29 [02:39<01:54,  6.03s/it]{'Model': 'GaussianNB', 'Accuracy': 0.9763877672834521, 'Balanced Accuracy': 0.9117748182559462, 'ROC AUC': 0.9117748182559462, 'F1 Score': 0.9865243411115635, 'recall_score': 0.8469387755102041, 'Time taken': 0.22774100303649902}
 38%|███▊      | 11/29 [02:44<01:43,  5.72s/it]{'Model': 'KNeighborsClassifier', 'Accuracy': 0.9995611109160493, 'Balanced Accuracy': 0.9030084670334088, 'ROC AUC': 0.9030084670334088, 'F1 Score': 0.9995455470408762, 'recall_score': 0.8061224489795918, 'Time taken': 5.023436069488525}

Best Models from LazyClassifier (comparing using recall_score)
AdaBoostClassifier
BaggingClassifier
DecisionTreeClassifier
ExtraTreeClassifier
ExtraTreesClassifier
GaussianNB (best performing)
KNeighborsClassifier


'Model': 'xgboost-train', 'recall_score': 1.00, 'f1_score': 1.00, 'balanced_accuracy_score': 1.00, 'roc_auc_score': 1.00, 'average_precision_score': 1.00
'Model': 'xgboost-test', 'recall_score': 0.82, 'f1_score': 0.86, 'balanced_accuracy_score': 0.91, 'roc_auc_score': 0.91, 'average_precision_score': 0.75

'Model': 'LinearSVC-train', 'recall_score': 0.74, 'f1_score': 0.81, 'balanced_accuracy_score': 0.87, 'roc_auc_score': 0.87, 'average_precision_score': 0.66
'Model': 'LinearSVC-test', 'recall_score': 0.78, 'f1_score': 0.80, 'balanced_accuracy_score': 0.89, 'roc_auc_score': 0.89, 'average_precision_score': 0.64

'Model': 'svc-train', 'recall_score': 0.81, 'f1_score': 0.89, 'balanced_accuracy_score': 0.90, 'roc_auc_score': 0.90, 'average_precision_score': 0.80
'Model': 'svc-test', 'recall_score': 0.67, 'f1_score': 0.79, 'balanced_accuracy_score': 0.84, 'roc_auc_score': 0.84, 'average_precision_score': 0.64

'Model': 'rf-train', 'recall_score': 1.00, 'f1_score': 1.00, 'balanced_accuracy_score': 1.00, 'roc_auc_score': 1.00, 'average_precision_score': 1.00
'Model': 'rf-test', 'recall_score': 0.81, 'f1_score': 0.86, 'balanced_accuracy_score': 0.90, 'roc_auc_score': 0.90, 'average_precision_score': 0.75

'Model': 'logistic_regression-train', 'recall_score': 0.64, 'f1_score': 0.75, 'balanced_accuracy_score': 0.82, 'roc_auc_score': 0.82, 'average_precision_score': 0.57
'Model': 'logistic_regression-test', 'recall_score': 0.65, 'f1_score': 0.73, 'balanced_accuracy_score': 0.83, 'roc_auc_score': 0.83, 'average_precision_score': 0.54

'Model': 'GradientBoosting-train', 'recall_score': 1.00, 'f1_score': 1.00, 'balanced_accuracy_score': 1.00, 'roc_auc_score': 1.00, 'average_precision_score': 1.00
'Model': 'GradientBoosting-test', 'recall_score': 0.78, 'f1_score': 0.76, 'balanced_accuracy_score': 0.89, 'roc_auc_score': 0.89, 'average_precision_score': 0.57

[Voting] .............. (3 of 3) Processing naive_bayes, total=   0.2s
[Voting] ...................... (1 of 3) Processing xgb, total=  51.3s
[Voting] ............ (2 of 3) Processing random_forest, total= 2.7min
'Model': 'voting_classifier-train', 'recall_score': 1.00, 'f1_score': 1.00, 'balanced_accuracy_score': 1.00, 'roc_auc_score': 1.00, 'average_precision_score': 1.00
'Model': 'voting_classifier-test', 'recall_score': 0.83, 'f1_score': 0.88, 'balanced_accuracy_score': 0.91, 'roc_auc_score': 0.91, 'average_precision_score': 0.77

