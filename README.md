# About this library

This library takes a list of strings and transforms each string into
a feature vector which is created from the concatenation of each character's
one hot vector.

Example:
Two strings "aa" and "ab" are ought to be transformed into vectors.
The allowed characters are "a" and "b".
The result after transformation for the list ["aa", "ab"] could then be
a matrix of the following structure:

| 1st char is "a" | 1st char is "b" | 2nd char is "a" | 2nd char is "b" |
| --------------- | --------------- | --------------- | --------------- |
| 1               | 0               | 1               | 0               |
| 1               | 0               | 0               | 1               |


Example usage in python:

```python
vectorizer = CharVectorizer("ab")
windows = [
    "aa",
    "ab"
]
target_length = max(len(window) for window in windows)
matrix = vectorizer.transform(windows, target_length)
```

This would transform each "a" to the one hot vector [1, 0] and each "b"
to [0, 1]. Therefore "aa" would be [1, 0, 1, 0] and "ab" [1, 0, 0, 1].
The result of the above example would then be a two dimensional numpy matrix,
containing both vectors as rows.

You may limit the characters to be transformed by changing the parameter in
the constructor. E.g. "ab" will (mostly) only accept the characters "a" and "b".
Other characters will by default get replaced by "#". You can change that
replacement by setting the parameter "map_unknown_chars_to" in the constructor.

As the vectorizer returns a matrix, all windows have to be equally sized.
Because of this, a target length for all strings has to be provided (as in
the above example). Too long strings will be shortened to that target length.
Too short strings will be increased in length by adding special characters
at the start or end of the string. Which special characters these are can by
defined in the constructor by setting the parameters "fill_right_char" or
"fill_left_char". By default both are " " (whitespace). In order to fill up
too short strings from the left, the method transform has to be called with
the parameter "fill_right=False". Otherwise they will be filled on the right.

# Examples

Simple example:

```python
from CharVectorizer import CharVectorizer

# Map only characters within the alphabet a-z.
# Capital letters will by default be mapped to lowercase letters.
vectorizer = CharVectorizer("abcdefghijklmnopqrstuvwxyz")

# The second string is shorter than the first one.
# A whitespace will be automatically added to the end.
windows = [
    "The fox jumped over the fence.",
    "The duck flew over the fence."
]
target_length = max(len(window) for window in windows)
# "matrix" is a numpy matrix.
# Each row resembles a string (so two rows).
matrix = vectorizer.transform(windows, target_length)
```

Reversing the transformation:

```python
vectorizer = CharVectorizer("abcdefghijklmnopqrstuvwxyz")
windows = ["foo", "bar", "FOO", "BAR", "?!"]
target_length = max(len(window) for window in windows)
matrix = vectorizer.transform(windows, target_length)

# Reverse vectorization
# The resulting list will contain:
#     ["foo", "bar", "foo", "bar", "## "]
# It doesn't contain FOO and BAR, because capital letters were not
# allowed and in that case (by default) those will be mapped to
# lowercase letters (if the respective lowercase letter is allowed).
# It doesn't contain "?!", but instead "## ", because "?" and "!"
# were not among the allowed characters and got replaced by the default
# replacement character (which is "#"). Additionally "?!" was too
# short, therefore it was extended in length (by default on the right,
# by default with whitespaces).
windows_rev = vectorizer.reverse_transform(matrix)
```

Changing the default replacement character for all characters that are not
allowed:

```python
vectorizer = CharVectorizer("abcdefghijklmnopqrstuvwxyz",
                            map_unknown_chars_to="?")
windows = ["Whoa."]
target_length = max(len(window) for window in windows)
matrix = vectorizer.transform(windows, target_length)
# Contains: ["whoa?"]
windows_rev = vectorizer.reverse_transform(matrix)
```

Filling too short strings on the left instead of the right:

```python
vectorizer = CharVectorizer("abcdefghijklmnopqrstuvwxyz")
windows = ["abc", "ab"]
target_length = max(len(window) for window in windows)
matrix = vectorizer.transform(windows, target_length, fill_right=False)
# Contains: ["abc", " ab"]
windows_rev = vectorizer.reverse_transform(matrix)
```

Changing the default filling characters:

```python
vectorizer = CharVectorizer("abcdefghijklmnopqrstuvwxyz",
                            fill_left_char=">",
                            fill_right_char="<")
windows = ["abc", "ab"]
target_length = max(len(window) for window in windows)
matrix1 = vectorizer.transform(windows, target_length)
matrix2 = vectorizer.transform(windows, target_length, fill_right=False)
# Contains: ["abc", "ab<"]
windows_rev1 = vectorizer.reverse_transform(matrix1)
# Contains: ["abc", ">ab"]
windows_rev2 = vectorizer.reverse_transform(matrix2)
```

Deactivating the automatic mapping of uppercase letters to lowercase ones:

```python
vectorizer = CharVectorizer("abcdefghijklmnopqrstuvwxyz",
                            auto_lowercase=False)

# The second string is shorter than the first one.
# A whitespace will be automatically added to the end.
windows = ["abc", "ABC"]
target_length = max(len(window) for window in windows)
matrix = vectorizer.transform(windows, target_length)
# contains: ["abc", "###"]
windows_rev = vectorizer.reverse_transform(matrix)
```

Activating the automatic mapping of lowercase to uppercase
letters (off by default):

```python
vectorizer = CharVectorizer("ABCDEFGHIJKLMNOPQRSTUVWXYZ",
                            auto_uppercase=True)

# The second string is shorter than the first one.
# A whitespace will be automatically added to the end.
windows = ["abc", "ABC"]
target_length = max(len(window) for window in windows)
matrix = vectorizer.transform(windows, target_length)
# contains: ["ABC", "ABC"]
windows_rev = vectorizer.reverse_transform(matrix)
```

Reversing vectors that do not only contain zeros and ones (e.g. output from
neural networks):

```python
# parameters here set for internal reasons to keep the example minimal
vectorizer = CharVectorizer("ab",
                            map_unknown_chars_to="a",
                            fill_left_char="a", fill_right_char="a")

matrix = numpy.array([[0.9, 0.1, 0.4, 0.6], [12, 5, 17, 6]])
# contains: ["ab", "aa"]
# maxval means "interpret the maximum value as a 1, all other components
# as 0" (for each one hot vector - notice that there are two per row here,
# because two characters are allowed)
windows_rev = vectorizer.reverse_transform_maxval(matrix)
```

# Performance

The library has been optimized to encode strings fast (python-fast, not c++-fast)
into vector representations. Using pure python to do that can easily be
two (or more) orders of magnitude slower.

The following code may be used to test the performance.
It generates randomly 10,000 strings with 50 characters each and then
transforms them 100 times into vector representations. In total
100*10000*50 = 50 million characters are transformed into vectors and
concatenated. The script ran in ~12 seconds on a Haswell 3.5 GHz cpu.

```python
from CharVectorizer import CharVectorizer
import random
import time

def create_random_text(chars, length):
    text = []
    max_l = len(chars)
    for _ in range(0, length):
        text.append(chars[random.randint(0, max_l - 1)])
    return "".join(text)

def create_random_texts(chars, length_per_string, count):
    result = []
    for _ in range(0, count):
        result.append(create_random_text(chars, length_per_string))
    return result

time_ms = lambda: int(round(time.time() * 1000))

vectorizer = CharVectorizer("abcdefghijklmnopqrstuvwxyz?!.,; ")
# build 10,000 random text windows of length 50 chars
# notice that these texts may contain digits, which are not allowed characters
# and will be replaced by "#"
texts = create_random_texts("abcdefghijklmnopqrstuvwxyz?!.,; 0123456789()",
                            50, 10000)

count = 0
t0 = time_ms()
# transform 1000 times
for _ in range(0, 100):
    matrix = vectorizer.transform(texts, 50)
    # add something from the matrix to make sure that nothing in the loop
    # is ever optimized away
    count += matrix[0][0]

# can be 0 if the first randomnly created string doesn't start with an "a"
print count

print "Required time: %f s" % ((time_ms() - t0) / float(1000))
```

# Requirements

Requires numpy.

# Tests

The library can be tested with

```
python TestCharVectorizer.py
```
