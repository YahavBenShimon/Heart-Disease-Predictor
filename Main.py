# Import the custom YBS_ML library. This library is assumed to contain functions for loading,
# normalizing data, and different classifier algorithms.
import YBS_ML as YBS

data, label = YBS.load_and_norm_data('heart.csv')

f1_log = YBS.log_classifier(data, label)
f1_KNN = YBS.KNN_classifier(data, label)
f1_RF = YBS.RF_classifier(data, label)
f1_SVM = YBS.SVM_classifier(data, label)
f1_ANN = YBS.ANN_classifier(data, label, num_of_ANN=5)

YBS.classfier_bar_plot(f1_log, f1_KNN, f1_RF, f1_ANN, f1_SVM)

