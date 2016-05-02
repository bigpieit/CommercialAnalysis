# CommercialAnalysis
The training.py provides the model selection details based on the data in "exam.dat.txt", including 
comparisons of:
1. k-nearest neighbor
2. logistic regression
3. support vector machin
  a. linear kernel
  b. radial basis function kernel
  c. polynomial kernel
  d. sigmoid kernel
The best one is selected according to it accuracy and receiver operating characteristic (ROC).

The testing.py implements the svm with radial basis function kernel(C = 4.7 , gamma = 1.1) to
predict new four feature vector. The input will be a four-feature like vector "f1 f2 f3 f4" or
a file like "exam.dat.txt"

Please read the attached report.pdf file for details of this work.
