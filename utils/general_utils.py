import pandas as pd
from sklearn.preprocessing import label_binarize


def check_for_complete_unique_attrs(cap_x_df: pd.DataFrame) -> list:

    print(f'the data frame has {cap_x_df.shape[0]} rows\n')

    concern_list = []
    for attr in cap_x_df.columns:
        label = ''
        if cap_x_df[attr].nunique() == cap_x_df.shape[0]:
            label = 'examine more closely'
            concern_list.append(attr)
        print(f'{attr} has {cap_x_df[attr].nunique()} unique values and is dtype {cap_x_df[attr].dtype} {label}')

    return concern_list


def label_binarize_(df: pd.DataFrame, target_attr: str, print_results: bool = True) -> (pd.DataFrame, dict):

    if df[target_attr].nunique() == 1:
        print(f'df[target_attr].nunique() = {df[target_attr].nunique()} - this case is not implemented')
        raise NotImplementedError()
    elif df[target_attr].nunique() == 2:
        df, lb_name_mapping = label_binarize_binary(df, target_attr, print_results=print_results)
    else:
        classes = df[target_attr].value_counts().index.tolist()
        num_label = [i for i in range(len(classes), 0, -1)]
        lb_name_mapping = dict(zip(classes, num_label))
        df[target_attr] = df[target_attr].map(lb_name_mapping)

    return df, lb_name_mapping


def label_binarize_binary(df: pd.DataFrame,
                          target_attr: str,
                          neg_label: int = 0,
                          pos_label: int = 1,
                          print_results: bool = True) -> (pd.DataFrame, dict):

    if print_results:
        print(f'\ndf[target_attr] is a string attribute')
        print(f'\ndf.loc[0:5, target_attr]:\n{df.loc[0:4, target_attr]}', sep='')
        print(f'\n{df[target_attr].value_counts(normalize=True)}')
        print(f'\nlabel encode df[target_attr]')

    # make more abundant class the negative label and the rarer class the positive label
    neg_str_label = df[target_attr].value_counts(normalize=True).idxmax()
    if df[target_attr].value_counts(normalize=True).idxmin() == neg_str_label:  # need to break tie
        label_list = df[target_attr].unique().tolist()
        label_list.remove(neg_str_label)
        pos_str_label = label_list[0]
    else:
        pos_str_label = df[target_attr].value_counts(normalize=True).idxmin()

    lb_name_mapping = {
        neg_label: neg_str_label,
        pos_label: pos_str_label
    }

    df.loc[:, target_attr] = label_binarize(
        df[target_attr],
        classes=[
            lb_name_mapping[0],
            lb_name_mapping[1]
        ],
        neg_label=neg_label,
        pos_label=pos_label,
        sparse_output=False
    )
    df[target_attr] = df[target_attr].astype(float)

    if print_results:
        print(f'\nafter label encoding df[target_attr]')
        print(f'\n{df[target_attr].value_counts(normalize=True)}')
        print(f'\ndf.loc[0:5, target_attr]:\n{df.loc[0:4, target_attr]}', sep='')
        print(f'\nlb_name_mapping: {lb_name_mapping}', sep='')

    return df, lb_name_mapping
