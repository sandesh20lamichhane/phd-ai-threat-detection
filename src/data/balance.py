"""Class balancing, applied ONLY to training folds, never to validation or test data."""


def balance_training(X_train, y_train, method="none", seed=0):
    if method == "none":
        return X_train, y_train
    from imblearn.over_sampling import SMOTE
    from imblearn.under_sampling import RandomUnderSampler

    sampler = {"smote": SMOTE(random_state=seed), "undersample": RandomUnderSampler(random_state=seed)}[method]
    return sampler.fit_resample(X_train, y_train)
