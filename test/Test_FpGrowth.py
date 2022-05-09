# this file contains a famous reference library for the fp growth algorithm. It will be used to test my own script.
import mlxtend.frequent_patterns
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from FpGrowth import get_data, fp_growth, count_items, sort_frequent_pattern_names, pretty_print_frequent_patterns
import deepdiff
from time import perf_counter


def get_mlxtend_result(dataset, min_support=0.5):

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


def test_own_algorithm(dataset: list[list], min_support=0.2):

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
    else:
        print(f'Everything works fine! Own time: {own_time*1e3:0.2f} ms. Reference time: {ref_time*1e3:0.2f} ms.')


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

    # test the datasets
    test_own_algorithm(dataset, min_support=0.000000001)
    test_own_algorithm(dataset2, min_support=0.000000001)
    test_own_algorithm(dataset3, min_support=0.000000001)


if __name__ == '__main__':
    main()
