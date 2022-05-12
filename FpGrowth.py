# this implements the FP growth algorithm
# it uses https://www.mygreatlearning.com/blog/understanding-fp-growth-algorithm/ as an example for values
from collections import Counter, OrderedDict, defaultdict
from itertools import combinations
from math import ceil
from typing import Union


def pythonic_count_items(table: list[list[str]]) -> [dict, int]:
    """
    This function creates a dictionary that contains counts for all the items in the transactions.

    :param table: list of transactions, where transaction is list of item
    :return: Returns a dict for all items
    """

    # flatten the two-dimensional list (and take care of double orders by using the count
    all_items = [item for sublist in table for item in sublist]

    # count occurrences and sort them by descending counter
    counter = Counter(all_items)

    return counter, len(table)


def fast_count_items(table: list[list[str]]) -> [dict, int]:
    """
    This function creates a dictionary that contains counts for all the items in the transactions. This function is
    slightly faster than the more pythonic version.

    :param table: list of transactions, where transaction is list of item
    :return: Returns a dict for all items
    """

    # create default dict with zeros as default
    counter = defaultdict(int)

    # iterate through the two-dimensional list
    for transaction in table:
        for item in transaction:
            counter[item] += 1

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


def create_sorted_representation(frequent_patterns: dict, counter: dict) -> dict:
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
    # comes with the cost of not being able to store attributes dynamically, but we don't need that anyway.
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

        # try to get the child from dict of children
        child = self.children.get(value)

        # create child and check singularity if we have more than one child after creation
        if child is None:

            # create child
            child = Node(self, value)

            # save child in dict of children
            self.children[value] = child

            # as tree can only go from singular -> not singular, we only need to check the children number if tree is
            # still singular.
            if self. singular and len(self.children) > 1:

                # set self and all others to not singular
                self.not_singular()

        # return the child node that has either been created or found in children dict
        return child

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
        # as the tree can only go from singular -> not singular, the first if saves functions calls in case we encounter
        # the first already not singular part of tree
        if self.singular:

            # set self to not singular
            self.singular = False

            # iteratively go up the tree
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


def construct_tree(table: list[list[str]], start_node_name: tuple = None, min_support=0):
    """
    This function creates a transaction tree for the fp growth algorithm.

    :param table: list of transactions.
    :param start_node_name: a name of the start node in order to support conditional trees.
    :param min_support: the given minimal support for the fp growth algorithm
    :return: the base node, the corresponding head table, the counter dict
    """

    # get the ordered counter for this table
    counter, number = fast_count_items(table)

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

    return base_node, head_table, counter


def count_frequent_patterns(table: list[Union[list, set]], condition: list = None, min_support=0,
                            frequent_patterns: dict = None):
    """
    This function recursively counts frequent patterns. It is able to support conditional trees.

    :param table: the table of transactions
    :param condition: the condition for the current tree
    :param min_support: the minimal support as int
    :param frequent_patterns: the dict of frequent patterns where the keys are frequent items and values are numbers
    :return: dict of frequent patterns with their support values
    """

    # build the first tree
    tree, head_table, counter = construct_tree(table=table, start_node_name=condition, min_support=min_support)

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
        for combination_length in range(1, len(node_list) + 1):

            # get the combinations
            combis = combinations(node_list, combination_length)

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

                # add combination to list and use frozen sets as keys as they are unordered and hashable
                # https://stackoverflow.com/questions/46633065/multiples-keys-dictionary-where-key-order-doesnt-matter
                frequent_patterns[frozenset(combi)] += support

    # recursively construct conditional trees if the current tree is not singular
    else:

        # build all conditional trees (conditions coming from the head table)
        for item in head_table:

            # initialize conditional transaction table
            conditional_table = []

            # iterate through all paths for one item and keep track of the condition support
            current_condition_support = 0
            for node in head_table[item]:

                # save the node counter of the lowest node in transaction path
                node_counter = node.counter

                # add the counter of the current transaction path to the overall support of the current condition
                current_condition_support += node_counter

                # initialize empty transaction
                transaction = []

                # get our parent node and check whether it is not root (parent node parents is not None)
                while node.parent.parent is not None:

                    # add parent node (higher support item) to our transaction at position zero to keep the transactions
                    # ordered
                    transaction.insert(0, node.parent.value)

                    # set our node to the parent node
                    node = node.parent

                # append the transaction n times (defined by last child node in one transaction)
                # as we are just counting we can just add references to our transaction multiple times to save memory
                if transaction:
                    conditional_table += [transaction] * node_counter

            # make the new condition for the conditional tree
            new_condition = condition + [item]

            # once we do construct a subtree, we need to add the condition to the counter and use frozen sets as keys as
            # they are unordered and hashable
            # https://stackoverflow.com/questions/46633065/multiples-keys-dictionary-where-key-order-doesnt-matter
            frequent_patterns[frozenset(new_condition)] += current_condition_support

            # call function recursively to build and traverse the next tree,
            # but now it is conditioned on our new condition
            _ = count_frequent_patterns(conditional_table,
                                        condition=new_condition,
                                        min_support=min_support,
                                        frequent_patterns=frequent_patterns)

    # sort the frequent patterns according to the counter (only if we are at highest level of recursion and therefore
    # have no condition
    if not condition:
        # sort the frequent patterns to correct string representation
        frequent_patterns = create_sorted_representation(frequent_patterns, counter)

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
    table = [set(transaction) for transaction in table]

    # start the frequent pattern counter
    frequent_patterns = count_frequent_patterns(table, min_support=ceil(len(table) * min_support))

    return frequent_patterns


def example_1():
    # data from the example in https://www.mygreatlearning.com/blog/understanding-fp-growth-algorithm/
    dataset = [['B', 'A', 'T'],
               ['A', 'C'],
               ['A', 'S'],
               ['B', 'A', 'C'],
               ['B', 'S'],
               ['A', 'S'],
               ['B', 'S'],
               ['B', 'A', 'S', 'T'],
               ['B', 'A', 'S']]

    # get the results
    result = fp_growth(dataset, min_support=0.2)

    # pretty print the results
    pretty_print_frequent_patterns(result, len(dataset))


def example_2():
    # data from http://rasbt.github.io/mlxtend/user_guide/frequent_patterns/fpgrowth/
    # Note the double onion buy in the last transaction
    dataset = [['Milk', 'Onion', 'Nutmeg', 'Kidney Beans', 'Eggs', 'Yogurt'],
               ['Dill', 'Onion', 'Nutmeg', 'Kidney Beans', 'Eggs', 'Yogurt'],
               ['Milk', 'Apple', 'Kidney Beans', 'Eggs'],
               ['Milk', 'Unicorn', 'Corn', 'Kidney Beans', 'Yogurt'],
               ['Corn', 'Onion', 'Onion', 'Kidney Beans', 'Ice cream', 'Eggs']]

    # get the results
    result = fp_growth(dataset, min_support=0.45)

    # pretty print the results
    pretty_print_frequent_patterns(result, len(dataset))


if __name__ == '__main__':
    example_1()
