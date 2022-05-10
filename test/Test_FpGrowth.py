# this file contains a famous reference library for the fp growth algorithm. It will be used to test my own script.
import mlxtend.frequent_patterns
from mlxtend.preprocessing import TransactionEncoder
from FpGrowth import get_data, fp_growth, count_items, sort_frequent_pattern_names, pretty_print_frequent_patterns
import deepdiff
from time import perf_counter
import tempfile
import sqlite3
import os.path
import urllib.request
import pandas as pd


def get_KDD_dataset():
    # Create a temporary directory
    dataset_folder = tempfile.mkdtemp()

    # Build path to database
    database_path = os.path.join(dataset_folder, "adventure-works.db")

    # Get the database
    urllib.request.urlretrieve(
        "https://github.com/FAU-CS6/KDD-Databases/raw/main/AdventureWorks/adventure-works.db",
        database_path,
    )

    # Open connection to the adventure-works.db
    connection = sqlite3.connect(database_path)

    order_df = pd.read_sql_query(
        "SELECT p.ProductID,p.Name,d.PurchaseOrderID,d.PurchaseOrderDetailID,d.ProductID "
        "FROM Product p "
        "JOIN PurchaseOrderDetail d ON p.ProductID = d.ProductID "
        "JOIN PurchaseOrderHeader h ON d.PurchaseOrderID = h.PurchaseOrderID ",
        connection,
        index_col="PurchaseOrderDetailID",
    )

    # get the unique values from the dataframe
    order_ids = order_df['PurchaseOrderID'].unique()

    # make a list of lists
    order_table = []
    for order_id in order_ids:
        order_table.append(list(order_df[order_df['PurchaseOrderID'] == order_id]['Name']))

    return order_table


def get_mlxtend_result(dataset, min_support=0.5):
    """
    Wrapper around the fp growth mlxtend implementation.

    :param dataset: the dataset of transactions
    :param min_support: the minimum support value
    :return: dictionary of frequent patterns, where the keys are item sets and the values are the support
    """

    #  make an assert statement about the support
    assert 0 <= min_support <= 1, f'[min_support] should be between 0 and 1. Currently it is [{min_support}].'

    # make the counter
    counter, number_of_transactions = count_items(dataset)

    # make the encoding
    te = TransactionEncoder()
    te_ary = te.fit(dataset).transform(dataset)

    # make the dataframe
    df = pd.DataFrame(te_ary, columns=te.columns_)

    result = mlxtend.frequent_patterns.fpgrowth(df, min_support=min_support, use_colnames=True)

    # make frequent pattern dict from the result
    frequent_patterns = {', '.join(pattern['itemsets']): int(pattern['support']*number_of_transactions)
                         for _, pattern in result.iterrows()}

    # sort the frequent patterns
    frequent_patterns = sort_frequent_pattern_names(frequent_patterns, counter)

    return frequent_patterns


def test_own_algorithm(dataset: list[list], min_support=0.2, verbose=False):
    """
    This function compares the output of the implementation given in this repository and a reference implementation from
    mlxtend (http://rasbt.github.io/mlxtend/user_guide/frequent_patterns/fpgrowth/).

    :param dataset: the dataset to compare the algorithms on
    :param min_support: the minimum support in percent
    :return: None
    """

    #  make an assert statement about the support
    assert 0 <= min_support <= 1, f'[min_support] should be between 0 and 1. Currently it is [{min_support}].'

    # get the result from own implementation
    timed = perf_counter()
    own_result = fp_growth(dataset, min_support=min_support)
    own_time = perf_counter() - timed

    # get the reference result
    timed = perf_counter()
    reference_result = get_mlxtend_result(dataset, min_support=min_support)
    ref_time = perf_counter() - timed

    # compare the two dicts
    if not reference_result == own_result:
        print('There is a difference between the results!')
        diff = deepdiff.DeepDiff(reference_result, own_result)
        print(diff.pretty())
        raise ValueError('Implementation and reference are not the same.')
    else:
        print(f'Everything works fine! Own time: {own_time*1e3:0.2f} ms. Reference time: {ref_time*1e3:0.2f} ms.')

    # make a nice print of the results if requested
    if verbose:
        print('\n')
        pretty_print_frequent_patterns(own_result, len(dataset))


def main():
    dataset = [['Milk', 'Onion', 'Nutmeg', 'Kidney Beans', 'Eggs', 'Yogurt'],
               ['Dill', 'Onion', 'Nutmeg', 'Kidney Beans', 'Eggs', 'Yogurt'],
               ['Milk', 'Apple', 'Kidney Beans', 'Eggs'],
               ['Milk', 'Unicorn', 'Corn', 'Kidney Beans', 'Yogurt'],
               ['Corn', 'Onion', 'Onion', 'Kidney Beans', 'Ice cream', 'Eggs']]
    dataset2 = get_data()
    dataset3 = [['Apple', 'Beer', 'Rice', 'Chicken'],
                ['Apple', 'Beer', 'Rice'],
                ['Apple', 'Beer'],
                ['Apple', 'Bananas'],
                ['Milk', 'Beer', 'Rice', 'Chicken'],
                ['Milk', 'Beer', 'Rice'],
                ['Milk', 'Beer'],
                ['Apple', 'Bananas']]
    dataset4 = get_KDD_dataset()

    # test the datasets
    test_own_algorithm(dataset, min_support=0.000000001)
    test_own_algorithm(dataset, min_support=0.60)
    test_own_algorithm(dataset2, min_support=0.000000001)
    test_own_algorithm(dataset3, min_support=0.000000001)
    test_own_algorithm(dataset4, min_support=0.02, verbose=True)


if __name__ == '__main__':
    main()
