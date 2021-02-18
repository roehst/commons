"""
A Pandas extension for Explorative Data Analysis.

Features allow the user to:
- Detect non-linear correlation in simple DataFrames
"""

import itertools

import pandas as pd
from pandas.api.extensions import register_dataframe_accessor
from sklearn.tree import DecisionTreeRegressor
from tabulate import tabulate


def _make_pairs(xs):
    return [(c1, c2) for (c1, c2) in itertools.product(xs, xs) if c1 != c2]


def _calculate_corr(x1, x2):
    x = x1.values.reshape(-1, 1)
    y = x2.values.reshape(-1, 1)
    model = DecisionTreeRegressor()
    model.fit(x, y)
    return model.score(x, y)


@register_dataframe_accessor("eda")
class EdaAccessor:
    def __init__(self, pandas_obj):
        self._obj = pandas_obj

    def correlations(self):

        numeric_columns = [
            column for column in self._obj.columns if self._obj.dtypes[column] == float
        ]

        rows = []

        rows.append(("column 1", "column 2", "score"))

        for column1, column2 in _make_pairs(numeric_columns):
            x1 = self._obj[column1]
            x2 = self._obj[column2]
            rows.append((column1, column2, _calculate_corr(x1, x2)))

        print(tabulate(rows))


if __name__ == "__main__":

    df = pd.read_csv(
        "https://forge.scilab.org/index.php/p/rdataset/source/file/master/csv/datasets/iris.csv"
    )

    df.eda.correlations()
