
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


some_function(3)
