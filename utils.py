
DATA_PATH = "Gene_Chip_Data/"
PCA_DATA_PATH = "./data/PCA/"
PCA_MODEL_PATH = "./model/PCA/"
LR_RES_PATH = "./result/LR/"
LSVM_RES_PATH = "./result/LSVM/"
SVM_RES_PATH = "./result/SVM/"
N_COMPONENTS = [0, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
C = [0.001, 0.01, 0.1, 1., 10., 100., 1000.]
KERNEL = ['linear', 'poly', 'rbf', 'sigmoid']