import pandas as pd
from sklearn.preprocessing import LabelBinarizer


def convert_categorical_to_on_hot_vec(df, categorical_feature):
    """
    This function convert categorical field into one hot vectors which will be concatenated in to the origin dataframe
    :param df: dataframe of the data
    :param categorical_feature: categorical feature name
    :return: ohv_df: hot vectors which will be concatenated into the origin dataframe
    """
    print("Start to convert attribute %s into one hot vector" % categorical_feature)
    jobs_encoder = LabelBinarizer ()
    jobs_encoder.fit(df[categorical_feature])
    transformed = jobs_encoder.transform (df[categorical_feature])
    ohv_df = pd.DataFrame(transformed)
    ohv_df.columns = [str(col)+'_' + categorical_feature for col in ohv_df.columns]
    print ("Finished converting attribute %s into one hot vector"%categorical_feature)
    return ohv_df

def pre_processing(df : pd.DataFrame):
    """
    This function is used to extract only the relevant columns and converting categorical field to hot vectors
    Nan values handling:
    1. Numeric attribute will be filled with mean of all values of this attriibute
    2. Categorical attribute will be filled with empty string, means new value for all Nan
    :param df: dataframe of the data
    :param file_name
    :return: df_new: preprocessed dataframe
    """

    #Replace TRUE or FALSE with number: 1 or 0 in accordance
    df.replace('TRUE', value=1, inplace=True)
    df.replace('FALSE', value=0, inplace=True)

    df_new = pd.DataFrame()
    categotial_columns = []
    numerical_columns = []
    columns_types = df.dtypes

    print("Start to handle NaN values:")
    for (column, type) in columns_types.items():
        if type == 'bool':
            df = df.astype({column: 'int64'})
        if type == 'object':
            categotial_columns.append(column)
            df[column].replace('','NAN', inplace=True)
            df[column] = df[column].astype('category')
            df[column] = df[column].cat.codes
        else:
            numerical_columns.append(column)
            mean = df[column].mean()
            df[column].fillna(mean,inplace=True)
    print("Finished handling NaN values.")

    print("Convert categorical attributes into one hot vectors:")
    for (column, type) in columns_types.items():
        if type == 'object':
            ohv_df = convert_categorical_to_on_hot_vec(df, column)
            df_new = pd.concat([df_new, ohv_df], axis=1)
        else:
            df_new[column] = df[column]
    print("Finished converting categorical attributes into one hot vectors.")


    #---------- start categories from zero
    min_value_class = df_new[df_new.columns[-1]].min()
    df_new[df_new.columns[-1]] = df_new[df_new.columns[-1]].add(-min_value_class)

    return df_new
