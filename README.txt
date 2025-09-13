Random Forest: 
statistical machine learning approach using statistical features such as mean, standard deviation, skewness, kurtosis among others. Also includes statistical tests such as:
  - Welch's t-test  (mean comparison, assuming unequal variance)
  - Levene's p-test (variance comparison)
  - Kolmogorov-Smirnov test  (general distribution comparison)
  - Mann-Whitney U test    (non-parametric test for median comparison)
Also factors in autocorrelation with lag 1, linear regression slope, and applying rolling windows to the statistical features (mean, std, etc)


Neural Network:
Very much a work in progress. The current architecture seems to be correct, since the same general approach of concatenating CNN (convolutional NN) and RNN (recurrent NN) outputs into a MLP appears in multiple research papers. However, performance is low due to what looks like overfitting, with ROC scores of validation batches always around 0.5 +- 0.05
CNN_RNN_Model tries this approach.
HybridModel attempts to include some statistical features, concatenating results with a simplified version of above CNN/RNN model, yet no improvement in performance.
