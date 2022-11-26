import pandas
import csv


class Dataset:

    def __init__(self, csv_file):
        """_summary_

        Args:
            cvs_file (): _description_
        """
        self.csv_file = csv_file

    def len(self):
        """_summary_

        Returns:
            _type_: _description_
        """
        with open(self.csv_file, 'rt') as f:
            return sum(1 for row in f) - 1

    def columns(self):
        with open(self.csv_file, 'rf') as f:
            columns = f.readline().rstrip().split(',')
        del columns[2]
        return columns

    def getitem(self, index):
        """
        Get example by index
        @param index: int
        @return: list, int
        """
        'Generates one sample of data'
        # Select sample
        idx = index + 1
        # Load data and get label
        with open(self.csv_file, 'rt') as f:
            reader = csv.reader(f)
            for line in reader:
                if str(idx) in line:
                    break

        y = int(line[2])
        del line[2]
        x = line
        return x, y

    def get_items(self, items_number):
        """
        Get specific amount of examples
        @param items_number:
        @return: pd.DataFrame, pd.Series
        """
        data = pd.read_csv(self.csv_file, nrows=items_number)
        y = data['Attrition_Flag']
        x = data.drop(['Attrition_Flag'], axis=1)
        return x, y
