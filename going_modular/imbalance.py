from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler


def balance_training_data(
    X_train,
    y_train,
    method="oversampling"
):
    """
    Balances only the training data.

    method:
        "oversampling"  -> increases the minority class
        "undersampling" -> reduces the majority class
        None            -> does not balance the data
    """

    print("Class distribution before balancing:")
    print(y_train.value_counts())

    if method == "oversampling":
        sampler = RandomOverSampler(random_state=42)

    elif method == "undersampling":
        sampler = RandomUnderSampler(random_state=42)

    elif method is None:
        return X_train, y_train

    else:
        raise ValueError(
            "method must be 'oversampling', "
            "'undersampling', or None"
        )

    X_train_balanced, y_train_balanced = sampler.fit_resample(
        X_train,
        y_train
    )

    print("\nClass distribution after balancing:")
    print(y_train_balanced.value_counts())

    return X_train_balanced, y_train_balanced