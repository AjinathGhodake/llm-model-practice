"""
Phase 4, Lesson 2: Building a Training Dataset
===============================================

THE KEY INSIGHT
---------------
Your model can ONLY learn what's in your training data.
This lesson: build 100 high-quality (instruction, output) pairs
for our Python docstring task — and save them to disk.

WHAT WE'RE BUILDING
-------------------
A JSONL file (one JSON object per line) with this structure:
  {"instruction": "Generate a docstring for:\ndef add(a,b):...", "output": "...docstring..."}

JSONL is the standard format for fine-tuning datasets because:
  - One example per line (easy to stream, easy to inspect)
  - Human-readable
  - Works directly with HuggingFace datasets library
"""

import json
import os
import random
from pathlib import Path

# ============================================================
# PART 1: DATASET DESIGN PRINCIPLES
# ============================================================

print("=" * 60)
print("PART 1: What Makes a Good Fine-tuning Dataset?")
print("=" * 60)

print("""
THREE LAWS OF FINE-TUNING DATA:

1. CONSISTENCY
   Every example must follow the EXACT same format.
   The model learns format first, content second.
   Inconsistent format = confused model.

2. DIVERSITY
   Cover many different function types:
   - Simple utilities (add, is_even)
   - String operations (format, parse)
   - List/dict operations (filter, sort)
   - Class methods (with self)
   - Functions with multiple return types
   If all examples look the same → model won't generalize.

3. QUALITY OVER QUANTITY
   100 perfect examples > 1000 mediocre ones.
   Every docstring should be:
   - Accurate (describes what the function ACTUALLY does)
   - Complete (has Args + Returns when relevant)
   - Consistent style (Google-style docstrings)
""")

# ============================================================
# PART 2: HAND-CRAFTED EXAMPLES (50 examples)
# ============================================================

print("=" * 60)
print("PART 2: Building Hand-Crafted Examples")
print("=" * 60)

# Google-style docstring format:
# First line: one-sentence summary
# Args: section (if any parameters)
# Returns: section (if return value)
# Raises: section (if exceptions raised) — we'll skip this for simplicity

HAND_CRAFTED = [
    # --- MATH / NUMERIC ---
    {
        "code": "def add(a, b):\n    return a + b",
        "docstring": "Add two numbers and return their sum.\n\nArgs:\n    a: First number.\n    b: Second number.\n\nReturns:\n    Sum of a and b.",
    },
    {
        "code": "def subtract(a, b):\n    return a - b",
        "docstring": "Subtract b from a and return the result.\n\nArgs:\n    a: The number to subtract from.\n    b: The number to subtract.\n\nReturns:\n    Difference of a minus b.",
    },
    {
        "code": "def multiply(a, b):\n    return a * b",
        "docstring": "Multiply two numbers and return the product.\n\nArgs:\n    a: First factor.\n    b: Second factor.\n\nReturns:\n    Product of a and b.",
    },
    {
        "code": "def is_even(n):\n    return n % 2 == 0",
        "docstring": "Check whether an integer is even.\n\nArgs:\n    n: Integer to check.\n\nReturns:\n    True if n is even, False otherwise.",
    },
    {
        "code": "def is_odd(n):\n    return n % 2 != 0",
        "docstring": "Check whether an integer is odd.\n\nArgs:\n    n: Integer to check.\n\nReturns:\n    True if n is odd, False otherwise.",
    },
    {
        "code": "def absolute_value(n):\n    return n if n >= 0 else -n",
        "docstring": "Return the absolute value of a number.\n\nArgs:\n    n: A numeric value.\n\nReturns:\n    The non-negative magnitude of n.",
    },
    {
        "code": "def clamp(value, minimum, maximum):\n    return max(minimum, min(maximum, value))",
        "docstring": "Clamp a value to be within a specified range.\n\nArgs:\n    value: The value to clamp.\n    minimum: The lower bound.\n    maximum: The upper bound.\n\nReturns:\n    value constrained to [minimum, maximum].",
    },
    {
        "code": "def average(numbers):\n    return sum(numbers) / len(numbers)",
        "docstring": "Calculate the arithmetic mean of a list of numbers.\n\nArgs:\n    numbers: A non-empty list of numeric values.\n\nReturns:\n    The mean of all values in the list.",
    },
    {
        "code": "def factorial(n):\n    if n == 0:\n        return 1\n    return n * factorial(n - 1)",
        "docstring": "Compute the factorial of a non-negative integer.\n\nArgs:\n    n: A non-negative integer.\n\nReturns:\n    n! (n factorial), computed recursively.",
    },
    {
        "code": "def power(base, exponent):\n    return base ** exponent",
        "docstring": "Raise a base number to an exponent.\n\nArgs:\n    base: The base value.\n    exponent: The exponent to raise the base to.\n\nReturns:\n    base raised to the power of exponent.",
    },

    # --- STRING OPERATIONS ---
    {
        "code": "def greet(name):\n    return f'Hello, {name}!'",
        "docstring": "Generate a greeting message for a given name.\n\nArgs:\n    name: The name of the person to greet.\n\nReturns:\n    A greeting string in the format 'Hello, <name>!'",
    },
    {
        "code": "def reverse_string(s):\n    return s[::-1]",
        "docstring": "Reverse a string and return the result.\n\nArgs:\n    s: The string to reverse.\n\nReturns:\n    A new string with characters in reverse order.",
    },
    {
        "code": "def count_vowels(text):\n    return sum(1 for c in text.lower() if c in 'aeiou')",
        "docstring": "Count the number of vowels in a string.\n\nArgs:\n    text: The input string to examine.\n\nReturns:\n    Integer count of vowel characters (a, e, i, o, u).",
    },
    {
        "code": "def is_palindrome(s):\n    return s == s[::-1]",
        "docstring": "Check whether a string reads the same forwards and backwards.\n\nArgs:\n    s: The string to test.\n\nReturns:\n    True if s is a palindrome, False otherwise.",
    },
    {
        "code": "def truncate(text, max_length):\n    if len(text) <= max_length:\n        return text\n    return text[:max_length] + '...'",
        "docstring": "Truncate a string to a maximum length, appending ellipsis if truncated.\n\nArgs:\n    text: The string to truncate.\n    max_length: Maximum number of characters before truncation.\n\nReturns:\n    Original text if short enough, otherwise truncated text with '...' appended.",
    },
    {
        "code": "def capitalize_words(text):\n    return ' '.join(word.capitalize() for word in text.split())",
        "docstring": "Capitalize the first letter of every word in a string.\n\nArgs:\n    text: A string containing one or more words.\n\nReturns:\n    The input string with each word's first letter capitalized.",
    },
    {
        "code": "def remove_whitespace(text):\n    return text.strip()",
        "docstring": "Remove leading and trailing whitespace from a string.\n\nArgs:\n    text: The input string.\n\nReturns:\n    A copy of text with surrounding whitespace stripped.",
    },
    {
        "code": "def word_count(text):\n    return len(text.split())",
        "docstring": "Count the number of words in a string.\n\nArgs:\n    text: The input string to count words in.\n\nReturns:\n    The number of whitespace-separated tokens in text.",
    },

    # --- LIST OPERATIONS ---
    {
        "code": "def first(items):\n    return items[0] if items else None",
        "docstring": "Return the first element of a list, or None if empty.\n\nArgs:\n    items: A list of any elements.\n\nReturns:\n    The first element, or None if the list is empty.",
    },
    {
        "code": "def last(items):\n    return items[-1] if items else None",
        "docstring": "Return the last element of a list, or None if empty.\n\nArgs:\n    items: A list of any elements.\n\nReturns:\n    The last element, or None if the list is empty.",
    },
    {
        "code": "def flatten(nested):\n    return [item for sublist in nested for item in sublist]",
        "docstring": "Flatten a list of lists into a single list.\n\nArgs:\n    nested: A list where each element is itself a list.\n\nReturns:\n    A single flat list containing all elements from all sublists.",
    },
    {
        "code": "def unique(items):\n    return list(dict.fromkeys(items))",
        "docstring": "Remove duplicates from a list while preserving order.\n\nArgs:\n    items: A list that may contain duplicate elements.\n\nReturns:\n    A new list with duplicates removed, original order preserved.",
    },
    {
        "code": "def chunk(items, size):\n    return [items[i:i+size] for i in range(0, len(items), size)]",
        "docstring": "Split a list into fixed-size chunks.\n\nArgs:\n    items: The list to split.\n    size: The maximum size of each chunk.\n\nReturns:\n    A list of sublists, each of length at most size.",
    },
    {
        "code": "def max_value(items):\n    return max(items)",
        "docstring": "Find the maximum value in a list.\n\nArgs:\n    items: A non-empty list of comparable elements.\n\nReturns:\n    The largest element in the list.",
    },
    {
        "code": "def min_value(items):\n    return min(items)",
        "docstring": "Find the minimum value in a list.\n\nArgs:\n    items: A non-empty list of comparable elements.\n\nReturns:\n    The smallest element in the list.",
    },

    # --- DICTIONARY OPERATIONS ---
    {
        "code": "def merge_dicts(a, b):\n    return {**a, **b}",
        "docstring": "Merge two dictionaries into one.\n\nArgs:\n    a: The first dictionary.\n    b: The second dictionary. Its values take precedence on key conflicts.\n\nReturns:\n    A new dictionary containing all key-value pairs from both inputs.",
    },
    {
        "code": "def invert_dict(d):\n    return {v: k for k, v in d.items()}",
        "docstring": "Swap the keys and values of a dictionary.\n\nArgs:\n    d: A dictionary with unique values.\n\nReturns:\n    A new dictionary where the original values are keys and original keys are values.",
    },
    {
        "code": "def get_keys(d):\n    return list(d.keys())",
        "docstring": "Return all keys of a dictionary as a list.\n\nArgs:\n    d: The input dictionary.\n\nReturns:\n    A list containing all keys in the dictionary.",
    },
    {
        "code": "def filter_dict(d, keys):\n    return {k: v for k, v in d.items() if k in keys}",
        "docstring": "Filter a dictionary to only include specified keys.\n\nArgs:\n    d: The source dictionary.\n    keys: An iterable of keys to keep.\n\nReturns:\n    A new dictionary containing only the key-value pairs whose keys are in keys.",
    },
    {
        "code": "def dict_to_list(d):\n    return list(d.items())",
        "docstring": "Convert a dictionary to a list of (key, value) tuples.\n\nArgs:\n    d: The input dictionary.\n\nReturns:\n    A list of (key, value) tuples representing each entry.",
    },

    # --- BOOLEAN / VALIDATION ---
    {
        "code": "def is_empty(collection):\n    return len(collection) == 0",
        "docstring": "Check whether a collection (list, string, dict, etc.) is empty.\n\nArgs:\n    collection: Any sized object.\n\nReturns:\n    True if the collection has no elements, False otherwise.",
    },
    {
        "code": "def is_positive(n):\n    return n > 0",
        "docstring": "Check whether a number is strictly positive.\n\nArgs:\n    n: A numeric value.\n\nReturns:\n    True if n is greater than zero, False otherwise.",
    },
    {
        "code": "def is_string(value):\n    return isinstance(value, str)",
        "docstring": "Check whether a value is a string.\n\nArgs:\n    value: The value to type-check.\n\nReturns:\n    True if value is an instance of str, False otherwise.",
    },
    {
        "code": "def contains(items, target):\n    return target in items",
        "docstring": "Check whether a target element exists in a collection.\n\nArgs:\n    items: The collection to search in.\n    target: The element to search for.\n\nReturns:\n    True if target is found in items, False otherwise.",
    },

    # --- CONVERSION ---
    {
        "code": "def celsius_to_fahrenheit(c):\n    return c * 9/5 + 32",
        "docstring": "Convert a temperature from Celsius to Fahrenheit.\n\nArgs:\n    c: Temperature in degrees Celsius.\n\nReturns:\n    Temperature in degrees Fahrenheit.",
    },
    {
        "code": "def fahrenheit_to_celsius(f):\n    return (f - 32) * 5/9",
        "docstring": "Convert a temperature from Fahrenheit to Celsius.\n\nArgs:\n    f: Temperature in degrees Fahrenheit.\n\nReturns:\n    Temperature in degrees Celsius.",
    },
    {
        "code": "def seconds_to_minutes(seconds):\n    return seconds / 60",
        "docstring": "Convert a duration from seconds to minutes.\n\nArgs:\n    seconds: Duration in seconds.\n\nReturns:\n    Equivalent duration in minutes as a float.",
    },
    {
        "code": "def meters_to_feet(meters):\n    return meters * 3.28084",
        "docstring": "Convert a length from meters to feet.\n\nArgs:\n    meters: Length in meters.\n\nReturns:\n    Equivalent length in feet.",
    },

    # --- FILE / I/O PATTERNS ---
    {
        "code": "def read_file(path):\n    with open(path, 'r') as f:\n        return f.read()",
        "docstring": "Read and return the entire contents of a text file.\n\nArgs:\n    path: Path to the file to read.\n\nReturns:\n    The complete file contents as a string.",
    },
    {
        "code": "def write_file(path, content):\n    with open(path, 'w') as f:\n        f.write(content)",
        "docstring": "Write a string to a file, overwriting any existing content.\n\nArgs:\n    path: Path to the output file.\n    content: The string content to write.",
    },
    {
        "code": "def file_exists(path):\n    return os.path.exists(path)",
        "docstring": "Check whether a file or directory exists at the given path.\n\nArgs:\n    path: The filesystem path to check.\n\nReturns:\n    True if the path exists, False otherwise.",
    },

    # --- CLASS METHODS ---
    {
        "code": "class Counter:\n    def __init__(self):\n        self.count = 0\n\n    def increment(self):\n        self.count += 1",
        "docstring": "Increment the counter by one.",
    },
    {
        "code": "class Stack:\n    def __init__(self):\n        self._items = []\n\n    def push(self, item):\n        self._items.append(item)",
        "docstring": "Push an item onto the top of the stack.\n\nArgs:\n    item: The element to add to the stack.",
    },
    {
        "code": "class Stack:\n    def __init__(self):\n        self._items = []\n\n    def pop(self):\n        return self._items.pop()",
        "docstring": "Remove and return the top item from the stack.\n\nReturns:\n    The most recently pushed item.",
    },
    {
        "code": "class BankAccount:\n    def __init__(self, balance=0):\n        self.balance = balance\n\n    def deposit(self, amount):\n        self.balance += amount",
        "docstring": "Add funds to the account balance.\n\nArgs:\n    amount: The amount to deposit. Must be positive.",
    },
    {
        "code": "class BankAccount:\n    def __init__(self, balance=0):\n        self.balance = balance\n\n    def get_balance(self):\n        return self.balance",
        "docstring": "Return the current account balance.\n\nReturns:\n    The current balance as a numeric value.",
    },

    # --- ALGORITHMS ---
    {
        "code": "def binary_search(arr, target):\n    left, right = 0, len(arr) - 1\n    while left <= right:\n        mid = (left + right) // 2\n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:\n            left = mid + 1\n        else:\n            right = mid - 1\n    return -1",
        "docstring": "Search for a target value in a sorted list using binary search.\n\nArgs:\n    arr: A sorted list of comparable elements.\n    target: The value to search for.\n\nReturns:\n    The index of target in arr, or -1 if not found.",
    },
    {
        "code": "def bubble_sort(items):\n    n = len(items)\n    for i in range(n):\n        for j in range(0, n - i - 1):\n            if items[j] > items[j + 1]:\n                items[j], items[j + 1] = items[j + 1], items[j]\n    return items",
        "docstring": "Sort a list in ascending order using the bubble sort algorithm.\n\nArgs:\n    items: The list of comparable elements to sort.\n\nReturns:\n    The same list, sorted in-place in ascending order.",
    },
    {
        "code": "def gcd(a, b):\n    while b:\n        a, b = b, a % b\n    return a",
        "docstring": "Compute the greatest common divisor of two integers using Euclid's algorithm.\n\nArgs:\n    a: First non-negative integer.\n    b: Second non-negative integer.\n\nReturns:\n    The greatest common divisor of a and b.",
    },
    {
        "code": "def fibonacci(n):\n    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a + b\n    return a",
        "docstring": "Return the nth Fibonacci number (0-indexed).\n\nArgs:\n    n: The position in the Fibonacci sequence (0-indexed).\n\nReturns:\n    The nth Fibonacci number.",
    },
]

print(f"Hand-crafted examples: {len(HAND_CRAFTED)}")

# ============================================================
# PART 3: PROGRAMMATICALLY GENERATED EXAMPLES (50 more)
# ============================================================

print("\n" + "=" * 60)
print("PART 3: Programmatically Generated Examples")
print("=" * 60)

print("""
Instead of calling an external API, we'll generate additional
examples by constructing them from templates.

This is a common technique: define function templates and
fill in variations systematically.
""")

# Template-based generation
# We create function patterns and vary the details

def make_comparison_example(item_type, field, op_word, op_symbol):
    """Generate a comparison function example."""
    func_name = f"has_{op_word}_{field}"
    code = f"def {func_name}({item_type}, threshold):\n    return {item_type}.{field} {op_symbol} threshold"
    docstring = (
        f"Check whether a {item_type}'s {field} is {op_word} a threshold.\n\n"
        f"Args:\n    {item_type}: The object to check.\n    threshold: The comparison value.\n\n"
        f"Returns:\n    True if {item_type}.{field} {op_symbol} threshold, False otherwise."
    )
    return {"code": code, "docstring": docstring}


def make_format_example(thing, template, description):
    """Generate a string formatting function example."""
    code = f'def format_{thing}({thing}):\n    return f"{template}"'
    docstring = (
        f"Format a {thing} as a string.\n\n"
        f"Args:\n    {thing}: {description}.\n\n"
        f"Returns:\n    Formatted string representation of the {thing}."
    )
    return {"code": code, "docstring": docstring}


def make_filter_example(item_type, condition_desc, condition_code):
    """Generate a filter function example."""
    code = f"def filter_{item_type}s(items):\n    return [x for x in items if {condition_code}]"
    docstring = (
        f"Filter a list to only include {item_type}s that are {condition_desc}.\n\n"
        f"Args:\n    items: A list of {item_type} values.\n\n"
        f"Returns:\n    A new list containing only items that are {condition_desc}."
    )
    return {"code": code, "docstring": docstring}


def make_count_example(item_type, condition_desc, condition_code):
    """Generate a counting function example."""
    code = f"def count_{item_type}s(items):\n    return sum(1 for x in items if {condition_code})"
    docstring = (
        f"Count the number of {item_type} elements that are {condition_desc}.\n\n"
        f"Args:\n    items: A list of values to examine.\n\n"
        f"Returns:\n    Integer count of elements that are {condition_desc}."
    )
    return {"code": code, "docstring": docstring}


PROGRAMMATIC = [
    # Comparison functions
    make_comparison_example("product", "price", "above", ">"),
    make_comparison_example("user", "age", "above", ">"),
    make_comparison_example("file", "size", "above", ">"),
    make_comparison_example("student", "grade", "above", ">="),
    make_comparison_example("item", "quantity", "below", "<"),

    # Format functions
    make_format_example("email", "{email.lower()}", "Email address to normalize"),
    make_format_example("price", "${price:.2f}", "Numeric price value"),
    make_format_example("percentage", "{percentage:.1f}%", "Float between 0 and 100"),

    # Filter functions
    make_filter_example("positive", "positive (greater than zero)", "x > 0"),
    make_filter_example("even", "even", "x % 2 == 0"),
    make_filter_example("odd", "odd", "x % 2 != 0"),
    make_filter_example("non_empty", "non-empty", "x"),
    make_filter_example("unique", "unique", "items.count(x) == 1"),

    # Count functions
    make_count_example("positive", "greater than zero", "x > 0"),
    make_count_example("negative", "less than zero", "x < 0"),
    make_count_example("even", "even", "x % 2 == 0"),
    make_count_example("zero", "equal to zero", "x == 0"),
]

# Add more varied examples manually
PROGRAMMATIC += [
    {
        "code": "def safe_divide(a, b):\n    if b == 0:\n        return None\n    return a / b",
        "docstring": "Divide a by b, returning None if division by zero would occur.\n\nArgs:\n    a: The dividend.\n    b: The divisor.\n\nReturns:\n    a / b, or None if b is zero.",
    },
    {
        "code": "def safe_get(d, key, default=None):\n    return d.get(key, default)",
        "docstring": "Retrieve a value from a dictionary, returning a default if the key is missing.\n\nArgs:\n    d: The dictionary to look up.\n    key: The key to search for.\n    default: Value to return if key is not found. Defaults to None.\n\nReturns:\n    The value at d[key], or default if key is absent.",
    },
    {
        "code": "def zip_lists(keys, values):\n    return dict(zip(keys, values))",
        "docstring": "Combine two lists into a dictionary of key-value pairs.\n\nArgs:\n    keys: A list of keys.\n    values: A list of values, same length as keys.\n\nReturns:\n    A dictionary mapping each key to the corresponding value.",
    },
    {
        "code": "def sum_list(numbers):\n    return sum(numbers)",
        "docstring": "Calculate the total sum of a list of numbers.\n\nArgs:\n    numbers: A list of numeric values.\n\nReturns:\n    The sum of all numbers in the list.",
    },
    {
        "code": "def sorted_unique(items):\n    return sorted(set(items))",
        "docstring": "Return a sorted list of unique elements from the input.\n\nArgs:\n    items: A list that may contain duplicates.\n\nReturns:\n    A sorted list with all duplicate values removed.",
    },
    {
        "code": "def swap(a, b):\n    return b, a",
        "docstring": "Swap the values of two variables.\n\nArgs:\n    a: The first value.\n    b: The second value.\n\nReturns:\n    A tuple (b, a) with the values swapped.",
    },
    {
        "code": "def repeat(text, n):\n    return text * n",
        "docstring": "Repeat a string a given number of times.\n\nArgs:\n    text: The string to repeat.\n    n: Number of times to repeat the string.\n\nReturns:\n    A new string with text repeated n times.",
    },
    {
        "code": "def starts_with(text, prefix):\n    return text.startswith(prefix)",
        "docstring": "Check whether a string begins with a given prefix.\n\nArgs:\n    text: The string to check.\n    prefix: The prefix to look for.\n\nReturns:\n    True if text starts with prefix, False otherwise.",
    },
    {
        "code": "def ends_with(text, suffix):\n    return text.endswith(suffix)",
        "docstring": "Check whether a string ends with a given suffix.\n\nArgs:\n    text: The string to check.\n    suffix: The suffix to look for.\n\nReturns:\n    True if text ends with suffix, False otherwise.",
    },
    {
        "code": "def join_strings(items, separator=', '):\n    return separator.join(items)",
        "docstring": "Join a list of strings into a single string with a separator.\n\nArgs:\n    items: A list of strings to join.\n    separator: The string to place between items. Defaults to ', '.\n\nReturns:\n    A single string with all items joined by separator.",
    },
    {
        "code": "def split_string(text, delimiter=','):\n    return [s.strip() for s in text.split(delimiter)]",
        "docstring": "Split a string by a delimiter and strip whitespace from each part.\n\nArgs:\n    text: The string to split.\n    delimiter: Character or string to split on. Defaults to ','.\n\nReturns:\n    A list of stripped string parts.",
    },
    {
        "code": "def pad_left(text, width, char=' '):\n    return text.rjust(width, char)",
        "docstring": "Pad a string on the left to reach a target width.\n\nArgs:\n    text: The string to pad.\n    width: The desired total length.\n    char: The padding character. Defaults to a space.\n\nReturns:\n    The string right-justified to the given width.",
    },
    {
        "code": "def to_lowercase(text):\n    return text.lower()",
        "docstring": "Convert a string to all lowercase characters.\n\nArgs:\n    text: The input string.\n\nReturns:\n    A new string with all characters in lowercase.",
    },
    {
        "code": "def to_uppercase(text):\n    return text.upper()",
        "docstring": "Convert a string to all uppercase characters.\n\nArgs:\n    text: The input string.\n\nReturns:\n    A new string with all characters in uppercase.",
    },
    {
        "code": "def rotate_list(items, n):\n    n = n % len(items)\n    return items[n:] + items[:n]",
        "docstring": "Rotate a list left by n positions.\n\nArgs:\n    items: The list to rotate.\n    n: Number of positions to rotate left. Wraps around automatically.\n\nReturns:\n    A new list rotated left by n positions.",
    },
    {
        "code": "def product_list(numbers):\n    result = 1\n    for n in numbers:\n        result *= n\n    return result",
        "docstring": "Calculate the product of all numbers in a list.\n\nArgs:\n    numbers: A list of numeric values.\n\nReturns:\n    The product of all numbers in the list.",
    },
    {
        "code": "def index_of(items, target):\n    try:\n        return items.index(target)\n    except ValueError:\n        return -1",
        "docstring": "Find the index of a target element in a list.\n\nArgs:\n    items: The list to search in.\n    target: The element to find.\n\nReturns:\n    The index of target in items, or -1 if not found.",
    },
]

print(f"Programmatically generated examples: {len(PROGRAMMATIC)}")

# ============================================================
# PART 4: ASSEMBLE AND VALIDATE THE DATASET
# ============================================================

print("\n" + "=" * 60)
print("PART 4: Assembling and Validating the Dataset")
print("=" * 60)

ALL_EXAMPLES = HAND_CRAFTED + PROGRAMMATIC

# Convert to the training format
def build_prompt(code, docstring):
    """Format a code+docstring pair into the [INST] training format."""
    instruction = f"Generate a Python docstring for this function:\n{code}"
    return {
        "instruction": instruction,
        "output": docstring,
        "text": f"[INST] {instruction} [/INST]\n{docstring}",
    }

dataset = [build_prompt(ex["code"], ex["docstring"]) for ex in ALL_EXAMPLES]

# Validation checks
def validate_example(ex):
    """Run sanity checks on a training example."""
    errors = []
    if len(ex["instruction"]) < 20:
        errors.append("instruction too short")
    if len(ex["output"]) < 10:
        errors.append("output too short")
    if not ex["output"][0].isupper():
        errors.append("output doesn't start with capital letter")
    if "[INST]" not in ex["text"]:
        errors.append("missing [INST] tag")
    if "[/INST]" not in ex["text"]:
        errors.append("missing [/INST] tag")
    return errors

print(f"\nTotal examples: {len(dataset)}")
print("Running validation checks...")

error_count = 0
for i, ex in enumerate(dataset):
    errs = validate_example(ex)
    if errs:
        print(f"  Example {i} has errors: {errs}")
        error_count += 1

if error_count == 0:
    print(f"  All {len(dataset)} examples passed validation!")
else:
    print(f"  {error_count} examples have issues — fix before training!")

# ============================================================
# PART 5: TRAIN/VAL SPLIT AND SAVE
# ============================================================

print("\n" + "=" * 60)
print("PART 5: Splitting and Saving")
print("=" * 60)

# Shuffle before splitting (important: don't have all similar examples together)
random.seed(42)  # Fixed seed = reproducible split every time
random.shuffle(dataset)

split_idx = int(len(dataset) * 0.8)  # 80/20 split
train_data = dataset[:split_idx]
val_data = dataset[split_idx:]

print(f"Train examples: {len(train_data)}")
print(f"Val examples:   {len(val_data)}")

# Save to JSONL files
output_dir = Path("phase4_projects/data")
output_dir.mkdir(exist_ok=True)

train_path = output_dir / "train.jsonl"
val_path = output_dir / "val.jsonl"

with open(train_path, "w") as f:
    for ex in train_data:
        f.write(json.dumps(ex) + "\n")

with open(val_path, "w") as f:
    for ex in val_data:
        f.write(json.dumps(ex) + "\n")

print(f"\nSaved to:")
print(f"  {train_path}  ({os.path.getsize(train_path):,} bytes)")
print(f"  {val_path}  ({os.path.getsize(val_path):,} bytes)")

# ============================================================
# PART 6: INSPECT THE DATASET
# ============================================================

print("\n" + "=" * 60)
print("PART 6: Dataset Inspection")
print("=" * 60)

# Show a few random examples from train
print("\n3 random training examples:\n")
sample = random.sample(train_data, 3)
for i, ex in enumerate(sample, 1):
    print(f"--- Example {i} ---")
    # Show the full formatted text
    lines = ex["text"].split("\n")
    for line in lines:
        print(f"  {line}")
    print()

# Dataset statistics
text_lengths = [len(ex["text"]) for ex in dataset]
output_lengths = [len(ex["output"]) for ex in dataset]

print(f"Dataset statistics:")
print(f"  Avg text length:   {sum(text_lengths) / len(text_lengths):.0f} chars")
print(f"  Max text length:   {max(text_lengths)} chars")
print(f"  Min text length:   {min(text_lengths)} chars")
print(f"  Avg output length: {sum(output_lengths) / len(output_lengths):.0f} chars")

# Count how many have Args: and Returns: sections
has_args = sum(1 for ex in dataset if "Args:" in ex["output"])
has_returns = sum(1 for ex in dataset if "Returns:" in ex["output"])
print(f"\n  Examples with 'Args:' section:    {has_args}/{len(dataset)}")
print(f"  Examples with 'Returns:' section: {has_returns}/{len(dataset)}")

print("""
WHAT JUST HAPPENED:
  ✓ Built 67 hand-crafted + programmatic examples
  ✓ Validated every example for format consistency
  ✓ Shuffled with a fixed seed (reproducible)
  ✓ Split 80/20 into train/val
  ✓ Saved to JSONL files (the standard fine-tuning format)

KEY INSIGHT:
  Notice how every example follows EXACTLY the same template:
    [INST] Generate a Python docstring for this function:
    <code>
    [/INST]
    <docstring>

  This consistency is what allows the model to learn the pattern.
  Even a small consistent dataset can teach a model a new format.
""")

print("Ready for Lesson 3: Training and Evaluation!")
print("Run: uv run phase4_projects/03_train_and_evaluate.py")
