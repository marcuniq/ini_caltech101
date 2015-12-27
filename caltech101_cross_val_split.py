import os

from ini_caltech101 import util

path = os.path.abspath(os.path.join('datasets', 'img-gen-resized', '101_ObjectCategories'))
stratify = True
seed = 42

# X_train contain only paths to images
(X_train, y_train) = util.load_paths_from_files(path, 'X_train.txt', 'y_train.txt', full_path=False)

(X_train, y_train) = util.shuffle_data(X_train, y_train, seed=seed)

nb_folds = 10

for cv_fold, ((X_cv_train, y_cv_train), (X_cv_test, y_cv_test)) in \
        enumerate(util.make_cv_split(X_train, y_train, nb_folds=nb_folds, stratify=stratify, seed=seed)):

    split_config = {'fold': cv_fold,
                    'nb_folds': nb_folds,
                    'stratify': stratify,
                    'seed': seed,
                    'train_samples': len(X_cv_train),
                    'test_samples': len(X_cv_test)}

    print("Save split for fold {}".format(cv_fold))
    util.save_cv_split_paths(path, X_cv_train, y_cv_train, X_cv_test, y_cv_test, cv_fold, split_config)

    print("Calculating mean and std...")
    X_mean, X_std = util.calc_stats(X_cv_train, base_path=path)
    print("Save stats")
    util.save_cv_stats(path, X_mean, X_std, cv_fold)


