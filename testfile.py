def some_function(index):
    while True:
        generate_file_query = input('Run import code to generate missing files? y/n: ').upper()
        try:
            generate_file_query in "YN"
        except ValueError:
            continue
        else:
            if generate_file_query == 'Y':
                if index in [0, 1]:
                    print('self._generate_missing_eigenvectors()')
                    return
                elif index == 2:
                    print('self._generate_missing_eigenvalues()')
                    return
                else:
                    print(f"Index of value {index} was called")
                    return
            elif generate_file_query == 'N':
                print("\nWill not generate files. Exiting...\n")
                exit(0)


# some_function(3)


class Fibonacci:

    def __init__(self, upper_limit=None, show_full_list=None):

        self.final_value = 0
        self.values_list = []

        if upper_limit is None:
            self.upper_limit = int(input("Enter the upper limit: "))
        else:
            self.upper_limit = upper_limit

        if show_full_list is None:
            self._test_show_list()
        else:
            self.show_full_list = show_full_list

    def _fibonacci(self, num):

        if num in [0, 1]:
            return num
        else:
            return self._fibonacci(num - 1) + self._fibonacci(num - 2)

    def _test_show_list(self):

        should_show_list = input("Print full sequence? Y/N: ").upper()
        try:
            while should_show_list not in "YN":
                should_show_list = input("Print full sequence? Y/N: ").upper()
        except ValueError:
            raise ValueError
        else:
            if should_show_list == 'Y':
                self.show_full_list = True
            elif should_show_list == 'N':
                self.show_full_list = False

    def generate_sequence(self):

        for current_iter in range(self.upper_limit):
            if self.show_full_list:
                self.values_list.append(self._fibonacci(current_iter))
            else:
                self.final_value = self._fibonacci(current_iter)

    def print_output(self):
        if self.show_full_list:
            print(self.values_list)
        else:
            print(self.final_value)


x = Fibonacci()
x.generate_sequence()
x.print_output()
