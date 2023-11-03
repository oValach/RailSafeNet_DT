def remove_and_renumber(numbers, to_remove):
    # Create a list of numbers to remove
    to_remove_set = set(to_remove)

    # Create a new list with the remaining numbers
    remaining_numbers = [num for num in numbers if num not in to_remove_set]
    print(remaining_numbers)
    # Renumber the remaining numbers from 0 to the count of remaining numbers
    renumbered_numbers = list(range(len(remaining_numbers)))

    return renumbered_numbers

# Example usage:
original_numbers = list(range(21))  # Create a list of numbers from 0 to 20
numbers_to_remove = [2, 5, 10]  # Numbers you want to remove

renumbered_result = remove_and_renumber(original_numbers, numbers_to_remove)
print(renumbered_result)
