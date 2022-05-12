# this file contains a famous reference library for the fp growth algorithm. It will be used to test my own script.
import mlxtend.frequent_patterns
from mlxtend.preprocessing import TransactionEncoder
from FpGrowth import fp_growth, count_items, pretty_print_frequent_patterns, create_sorted_representation
import deepdiff
from time import perf_counter
import tempfile
import sqlite3
import os.path
import urllib.request
import pandas as pd
import requests
import re
import fpgrowth_py
from csv import reader


def get_from_csv(path='data.csv') -> list[list[str]]:
    """
    Function to load a csv file as a dataset
    :param path: Path to the csv file
    :return: the dataset in as a list of transactions where each transaction in a list of items as strings
    """
    data = []
    with open(path, 'r') as file:
        csv_reader = reader(file)
        for line in csv_reader:
            line = list(filter(None, line))
            data.append(line)

    return data


def get_data() -> list[list]:
    """
    This function can be used to extract transaction data from a website if the data is contained in a standard html
    table with two columns (transcation ID, item list). The item list needs to be comma separated. The table should have
    n+1 rows. Where the first row (+1) is are the column names and the other n are n transactions.

    :return: a transaction list, where the second level lists (list[**LISTS**]) is a list of items and the first level
    list is the list of transactions.
    """

    # make a header
    headers = {'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko)'
                             ' Chrome/75.0.3770.80 Safari/537.36'}

    # get the website text
    response = requests.get('https://www.mygreatlearning.com/blog/understanding-fp-growth-algorithm/', headers=headers)

    # find all tables
    table = re.findall(r'<tbody>.*</tbody>',
                       response.text)
    # check whether is has found all expected tables
    assert len(table) == 4, 'There has been something wrong with parsing the website.'

    # find all rows
    table = re.findall(r'<tr>.*?</tr>', table[0])

    # delete first row
    table = table[1:]

    # find all columns
    column_re = re.compile(r'<td>.*?</td>')
    table = [[column[4:-5].replace(' ', '').split(',') for column in re.findall(column_re, row)] for row in table]

    # get rid of transaction names
    table = [row[1] for row in table]

    return table


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

    # make the encoding
    te = TransactionEncoder()
    te_ary = te.fit(dataset).transform(dataset)

    # make the dataframe
    df = pd.DataFrame(te_ary, columns=te.columns_)

    result = mlxtend.frequent_patterns.fpgrowth(df, min_support=min_support, use_colnames=True)

    # make the counter
    counter, number_of_transactions = count_items([list(set(transaction)) for transaction in dataset])
    # make frequent pattern dict from the result
    frequent_patterns = {pattern['itemsets']: round(pattern['support'] * number_of_transactions)
                         for _, pattern in result.iterrows()}

    # sort the frequent patterns
    frequent_patterns = create_sorted_representation(frequent_patterns, counter)

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

    # get the reference result -----------------------------------------------------------------------------------------
    timed = perf_counter()
    reference_result = get_mlxtend_result(dataset, min_support=min_support)
    ref_time = perf_counter() - timed

    # get the result from own implementation ---------------------------------------------------------------------------
    timed = perf_counter()
    own_result = fp_growth(dataset, min_support=min_support)
    own_time = perf_counter() - timed

    # get the result from Medium package -------------------------------------------------------------------------------
    # https://towardsdatascience.com/fp-growth-frequent-pattern-generation-in-data-mining-with-python-implementation-244e561ab1c3
    # https://github.com/chonyy/fpgrowth_py
    timed = perf_counter()
    second_result, _ = fpgrowth_py.fpgrowth(dataset, minSupRatio=min_support, minConf=0)
    second_time = perf_counter() - timed

    # compare reference with own implementation ------------------------------------------------------------------------
    if not reference_result == own_result:
        print('There is a difference between the results! The print goes from reference to own.')
        diff = deepdiff.DeepDiff(reference_result, own_result)
        print(diff.pretty())
        print(f'Own time: {own_time*1e3:0.2f} ms. Reference time: {ref_time*1e3:0.2f} ms.')
        raise ValueError('Implementation and reference are not the same.')
    else:
        print(f'Everything works fine! Own time: {own_time*1e3:0.2f} ms. Reference time: {ref_time*1e3:0.2f} ms.'
              f' Second time: {second_time*1e3:0.2f} ms')

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
    dataset5 = get_from_csv()

    # make a print to see when tests are started
    print('Tests will be starting: \n')

    # test the datasets
    test_own_algorithm(dataset, min_support=0.000000001)
    test_own_algorithm(dataset, min_support=0.60)
    test_own_algorithm(dataset2, min_support=0.000000001)
    test_own_algorithm(dataset2, min_support=0.7)
    test_own_algorithm(dataset3, min_support=0.000000001)
    test_own_algorithm(dataset3, min_support=0.2)
    test_own_algorithm(dataset4, min_support=0.02, verbose=True)
    test_own_algorithm(dataset4, min_support=0.005)
    test_own_algorithm(dataset5, min_support=0.15)


if __name__ == '__main__':
    main()
