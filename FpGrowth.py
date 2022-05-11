# this implements the FP growth algorithm
# it uses https://www.mygreatlearning.com/blog/understanding-fp-growth-algorithm/ as an example for values
import requests
import re
from collections import Counter, OrderedDict, defaultdict
from itertools import combinations
from math import ceil


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


def count_items(table: list[list[str]]) -> [OrderedDict, int]:
    """
    This function creates a dictionary that contains counts for all the items in the transactions.

    :param table: list of transactions
    :return: Returns an ordered dict for all items in descending item count order
    """

    # flatten the two-dimensional list (and take care of double orders by using the count
    all_items = [item for sublist in table for item in sublist]

    # count occurrences and sort them by descending counter
    counter = OrderedDict(Counter(all_items).most_common())

    return counter, len(table)


def sort_transactions(table: list[list[str]], counter: dict) -> list[list[str]]:
    """
    This function sort a table of transactions according to the counter.
    :param table: the list of transactions.
    :param counter: the counter for every item in the transactions.
    :return: Sorted table of transactions.
    """
    return [sort_items(transaction, counter) for transaction in table]


def sort_items(transaction: list[str], counter: dict) -> list[str]:
    """
    This functions sorts a list of items in the transaction according to the counter and their names (in order to
    keep from ambiguous sorting if the counter of two items is equal.
    :param transaction: list of items as strings
    :param counter: a dictionary that holds the number of items in the transaction
    :return: sorted list of items in descending amount
    """
    return sorted(transaction, key=lambda x: (counter[x], str(x)), reverse=True)


def delete_items_with_no_support(table: list[list[str]], counter: dict, min_support: int)\
        -> (list[list[str]], OrderedDict):
    """
    This function deletes transactions from the list of transactions if their support is not big enough.

    :param table: a two-dimensional list of transactions where each transaction is a list of times
    :param counter: a dictionary that counts the number of participations of each item
    :param min_support: the minimum support used for this dataset
    :return: the cleaned list of transcations
    """
    # delete non frequent items from counter
    counter = OrderedDict([(item, number) for item, number in counter.items() if number >= min_support])

    # delete items if necessary
    new_table = [[item for item in transaction if counter.get(item)] for transaction in table]

    return new_table, counter


def sort_frequent_pattern_names(frequent_patterns: dict, counter: dict) -> dict:
    """
    This function sort the frequent pattern names according to their counter
    :param frequent_patterns: a dict with frequent patterns as key and support as value
    :param counter: a dict with the absolute counts of the items in the transaction table
    :return: the frequent pattern dict but with the names in order
    """
    frequent_patterns = {'; '.join(sort_items(list(name), counter)): value for name, value in frequent_patterns.items()}
    return frequent_patterns


class Node(object):

    # add slots to impede the creation of object __dict__ in order to reduce the memory footprint of each tree
    # comes with the cost du not store attributes dynamically, but we don't need that anyway.
    __slots__ = 'parent', 'value', 'children', 'counter', 'singular'

    def __init__(self, parent, value):
        """
        This function creates a base node for a tree. Please use base_node.adopt() for child creation!

        :param parent: the parent node in the tree.
        :param value: the value of the current node.
        """

        # save the values
        self.parent = parent  # the parent node (to traverse the tree)
        self.value = value  # the actual value (item) of the node
        self.children = {}  # a dict of children, where we will save them with their names
        self.counter = 0  # the counter how many times the item exists in the tree
        self.singular = True  # value to store whether the tree is singular (only one branch)

    def adopt(self, value):
        """
        This function creates a child node with the given value and inserts it in the tree. It also takes care of the
        tracking, whether the tree is singular.

        :param value: the item name
        :return: the child node
        """
        if value not in self.children:
            self.children[value] = Node(self, value)

        # go through tree and check if it is singular (no splits other than at root)
        if len(self.children) > 1:
            self.not_singular()

        return self.children[value]

    def increment(self):
        """
        This function recursively increments the use counter for all nodes in one branch.

        :return: None
        """
        self.counter += 1
        if self.parent is not None:
            self.parent.increment()

    def not_singular(self):
        """
        This function recursively sets all parent nodes to not singular once it is called.

        :return: None
        """
        self.singular = False
        if self.parent is not None:
            self.parent.not_singular()

    def pretty_print(self, heading='1.', start_str=""):
        """
        This function creates a nicely formatted string of the current node and it's childs.

        :param heading: The starting heading for the representation.
        :param start_str: The appendix for every line of the tree.
        :return: String representation of the tree.
        """
        tabs = "\t" * (len(heading) // 2 - 1)
        start_str += f'{tabs}{heading} {self.value}: {self.counter}\n'

        # make comment if tree is singular
        if self.parent is None:
            start_str = start_str[:-1] + (' -> singular\n' if self.singular else ' -> not singular\n')

        for counter, child in enumerate(self.children.values(), start=1):
            start_str += child.pretty_print(heading=heading+f'{str(counter)}.')

        return start_str

    def __str__(self):
        """
        This function is a wrapper so self.pretty_print() is called once somebody attempts to print the node.

        :return: Formatted string representation of the tree.
        """
        return self.pretty_print()


def construct_tree(table: list[list[str]], start_node_name: tuple = None, condition_support=0, min_support=0):
    """
    This function creates a transaction tree for the fp growth algorithm.

    :param table: list of transactions.
    :param start_node_name: a name of the start node in order to support conditional trees.
    :param condition_support: the support of the current condition for conditional trees.
    :param min_support: the given minimal support for the fp growth algorithm
    :return: the base node, the corresponding head table, the counter dict
    """

    # get the ordered counter for this table
    counter, number = count_items(table)

    # delete items with low support form the transactions
    table, counter = delete_items_with_no_support(table, counter, min_support)

    # make the head table to search for the nodes later
    head_table = {name: set() for name in counter}

    # sort the transactions if we build the first base tree. Otherwise, they will be sorted anyway from before.
    if start_node_name is None:
        table = sort_transactions(table, counter)

    # create the base node and the condition support
    base_node = Node(None, start_node_name)

    # construct the tree
    for transaction in table:

        # reset node to base node for next interaction
        current_node = base_node

        # go through one transaction
        for item in transaction:

            # adopt node in tree
            current_node = current_node.adopt(item)

            # put node into the head table for later referencing
            head_table[item].update([current_node])

        # increment the counters of nodes in transaction branch after transaction is completed
        current_node.increment()

    # include this check if you want to test the tree building in the algorithm. Otherwise, keep it commented.
    """
    # get a count of the current table
    counter, number = count_items(table)
    
    # check all nodes and convert set to list. We need it to be a set in the first place to not include nodes several
    # times. For return statement it needs to be a list of nodes, so we can index those nodes.
    for value, count in counter.items():
        if count >= min_support:
            # check all item nodes
            assert sum([node.counter for node in head_table[value]]) == count, \
                f'Item {value} has not the right amount in tree.'

            # check base node
            assert base_node.counter == len(table), 'Not all transactions have been build in the tree.'

            # convert set of nodes to list
            head_table[value] = list(head_table[value])
    """

    return base_node, head_table, counter


def count_frequent_patterns(table: list[list], condition: list = None, condition_support=0, min_support=0,
                            frequent_patterns: dict = None):
    """
    This function recursively counts frequent patterns. It is able to support conditional trees.

    :param table: the table of transactions
    :param condition: the condition for the current tree
    :param condition_support: the support of the current condition
    :param min_support: the minimal support as int
    :param frequent_patterns: the dict of frequent patterns where the keys are frequent items and values are numbers
    :return: dict of frequent patterns with their support values
    """

    # build the first tree
    tree, head_table, counter = construct_tree(table, condition, condition_support, min_support=min_support)

    # dict to save the frequent patterns
    if frequent_patterns is None:
        frequent_patterns = defaultdict(int)

    # make the condition
    if condition is None:
        condition = []

    # look if the tree is singular and if it is make the frequent pairs
    if tree.singular:

        # make a list of nodes
        node_list = list(head_table.keys())

        # make all the combinations from the child nodes in the conditional tree
        for n in range(1, len(node_list) + 1):

            # get the combinations
            combis = combinations(node_list, n)

            # go through all combinations and look for the lowest number as support value
            for combi in combis:

                # the maximum support we can handle is the amount of accesses to the root node
                # reset the lowest value to root node counter
                support = tree.counter
                for item in combi:
                    support = min(support, sum([node.counter for node in head_table[item]]))

                # build the combination as key for the dict
                if tree.value is not None:

                    # add condition to the item combination
                    combi = tree.value + list(combi)

                # add combination to list and use frozensets as keys as they are unordered and hashable
                # https://stackoverflow.com/questions/46633065/multiples-keys-dictionary-where-key-order-doesnt-matter
                frequent_patterns[frozenset(combi)] += support

    # recursively construct conditional trees if the current tree is not singular
    else:

        # go through all items in the header table
        for item in head_table:

            # make conditional transactions
            conditional_table = []

            # iterate through all paths for one item and keep track of the condition support
            current_condition_support = sum([node.counter for node in head_table[item]])

            for node in head_table[item]:

                # save the node counter of the lowest node in transaction path
                node_counter = node.counter

                # build one transaction
                transaction = []

                # get our parent node and check whether it is not root (parent node parents is not None)
                while node.parent.parent is not None:
                    transaction.insert(0, node.parent.value)
                    node = node.parent

                # append the transaction n times (defined by last child node in one transaction)
                # also make sure the transaction gets deep copied (list[:])
                if transaction:
                    conditional_table += [transaction] * node_counter

            # sort the new condition
            # new_condition = sort_items(condition + [item], base_counter)
            new_condition = condition + [item]

            # once we do construct a subtree, we need to add the condition to the counter and use frozensets as keys as
            # they are unordered and hashable
            # https://stackoverflow.com/questions/46633065/multiples-keys-dictionary-where-key-order-doesnt-matter
            frequent_patterns[frozenset(new_condition)] += current_condition_support

            # call function recursively to build and traverse the next tree,
            # but now it is conditioned on certain items with a certain support
            _ = count_frequent_patterns(conditional_table,
                                        condition=new_condition,
                                        condition_support=current_condition_support,
                                        min_support=min_support,
                                        frequent_patterns=frequent_patterns)

    # sort the frequent patterns according to the counter (only if we are at highest level of recursion and therefore
    # have no condition
    if not condition:
        # sort the frequent patterns to correct string representation
        frequent_patterns = sort_frequent_pattern_names(frequent_patterns, counter)

    return frequent_patterns


def pretty_print_frequent_patterns(frequent_patterns: dict, number_of_transactions: int, percentage_precision=2,
                                   sorted_print=True):
    """
    This function prints a human-readable version of the frequent patterns from the fp growth algorithm.

    :param frequent_patterns: a dict of frequent patterns where the values are their support.
    :param number_of_transactions: the total number of transactions as int.
    :param percentage_precision: the amount of decimals for the percentage value
    :param sorted_print: boolean value to activate sorted printing according to the support (descending)
    :return: None
    """

    # check for empty input
    if not frequent_patterns:
        print('No frequent patterns found.')
        return

    # sort the frequent patterns if required
    if sorted_print:
        frequent_patterns = OrderedDict(sorted(frequent_patterns.items(), key=lambda x: x[1], reverse=True))

    # save the column names
    columns = ['Pattern', 'Support', 'Support (%)']

    # get the string representation of all frequent patterns
    representations = [len(str(value)) for patterns in frequent_patterns.items() for value in patterns]

    # get the maximum length of counters and names
    max_length_names = max(representations[::2])
    max_length_counter = max(representations[1::2])

    # check the column length for pattern and counter
    max_length_names = max(max_length_names, len(columns[0]))
    max_length_counter = max(max_length_counter, len(columns[1]))

    # make the format string
    percentage_format = f'0.{percentage_precision}%'
    max_length_percent = max(len(f'{0.5:{percentage_format}}'), len(columns[2]))
    percentage_format = f'{max_length_percent}.{percentage_format[2:]}'

    # make the header of the table
    filler_string = '|-' + '-' * max_length_names + '-+-' + '-' * max_length_counter + '-+-'\
                    + '-' * max_length_percent + '-|\n'
    print_string = filler_string
    print_string += f'| {columns[0]:<{max_length_names}} | ' \
                    f'{columns[1]:<{max_length_counter}} | ' \
                    f'{columns[2]:<{max_length_percent}} |\n'
    print_string += filler_string

    # make all the rows
    for name, counter in frequent_patterns.items():
        print_string += f'| {name: >{max_length_names}} | ' \
                        f'{counter: >{max_length_counter}} | ' \
                        f'{counter/number_of_transactions:>{percentage_format}} |\n'

    # make the end row
    print_string += filler_string

    # print everything
    print(print_string)


def fp_growth(table: list[list[str]], min_support=0.5):
    """
    This function implements the fp growth algorithm. The items in the transactions need be given as a string
    representation!

    :param table: a list of transactions (list of lists, where the second level list is a list of items per transation)
    :param min_support: the minimum support for frequent patters in percentage (between 0 and 1)
    :return: a dict of frequent patterns
    """

    # make an assert statement about the support
    assert 0 <= min_support <= 1, f'[min_support] should be between 0 and 1. Currently it is [{min_support}].'

    # check the given dataset
    for trans_counter, transaction in enumerate(table):
        assert isinstance(transaction, list), f'The input [table] needs to be a list of transactions.' \
                                              f' Current type: {type(transaction)} for table[{trans_counter}].'
        for item_counter, item in enumerate(transaction):
            assert isinstance(item, str), f'The items in each transaction need to to be strings.' \
                                          f' Current type: {type(item)} for table[{trans_counter}][{item_counter}].'

    # take care of double orders in the table
    table = [list(set(transaction)) for transaction in table]

    # start the frequent pattern counter
    frequent_patterns = count_frequent_patterns(table, min_support=ceil(len(table) * min_support))

    return frequent_patterns


def example_use():
    # get the data from the website
    table = get_data()
    
    # get the results
    res = fp_growth(table, 0.33)

    # print the frequent patterns in a readable way
    pretty_print_frequent_patterns(res, len(table))


def example_use2():
    dataset = [['Milk', 'Onion', 'Nutmeg', 'Kidney Beans', 'Eggs', 'Yogurt'],
               ['Dill', 'Onion', 'Nutmeg', 'Kidney Beans', 'Eggs', 'Yogurt'],
               ['Milk', 'Apple', 'Kidney Beans', 'Eggs'],
               ['Milk', 'Unicorn', 'Corn', 'Kidney Beans', 'Yogurt'],
               ['Corn', 'Onion', 'Onion', 'Kidney Beans', 'Ice cream', 'Eggs']]

    # get the results
    result = fp_growth(dataset, min_support=0.2)

    # pretty print the results
    pretty_print_frequent_patterns(result, len(dataset))


if __name__ == '__main__':
    example_use()
    # example_use2()
