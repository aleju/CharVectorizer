import unittest
import random
import numpy as np
from CharVectorizer import CharVectorizer

ALPHABET_LOWERCASE = "abcdefghijklmnopqrstuvwxyz"
ALPHABET_UPPERCASE = "abcdefghijklmnopqrstuvwxyz".upper()
#PUNCTUATION = ".,:;?!-()"
DIGITS = "0123456789"

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

class TestCharVectorizer(unittest.TestCase):
    def test_sanity(self):
        count_chars = 20
        count_texts = 10
        vectorizer = CharVectorizer(ALPHABET_LOWERCASE)
        texts = create_random_texts(ALPHABET_LOWERCASE, count_chars,
                                    count_texts)
        matrix = vectorizer.transform(texts, count_chars)
        ones = 0
        zeros = 0
        for cell in np.nditer(matrix):
            if cell == 1:
                ones += 1
            else:
                zeros += 1
        
        self.assertEqual(ones, count_chars * count_texts)
        self.assertEqual(ones + zeros,
                        vectorizer.get_one_char_vector_length()
                        * count_chars * count_texts)
    
    def test_transform(self):
        # 1000 texts only with only known chars
        count_chars = 20
        count_texts = 1000
        vectorizer = CharVectorizer(ALPHABET_LOWERCASE)
        texts = create_random_texts(ALPHABET_LOWERCASE, count_chars,
                                    count_texts)
        matrix = vectorizer.transform(texts, count_chars)
        reverse_transformed = vectorizer.reverse_transform(matrix)
        for text_is, text_exp in zip(reverse_transformed, texts):
            self.assertEqual(text_is, text_exp)
        
        # 1000 texts only with only unknown chars
        count_chars = 20
        count_texts = 1000
        vectorizer = CharVectorizer(ALPHABET_LOWERCASE,
                                    map_unknown_chars_to="X")
        texts = create_random_texts(DIGITS, count_chars,
                                    count_texts)
        expected_str = "X" * count_chars
        matrix = vectorizer.transform(texts, count_chars)
        reverse_transformed = vectorizer.reverse_transform(matrix)
        for text_is, text_exp in zip(reverse_transformed, texts):
            self.assertEqual(text_is, expected_str)
        
        # 1000 texts with 20% unknown chars
        count_chars = 100
        count_texts = 1000
        map_unknown_chars_to = "X"
        known_chars = "abcdefghijklmnopqrstuvwx"
        unknown_chars = "!?&/()"
        vectorizer = CharVectorizer(known_chars,
                                    map_unknown_chars_to=map_unknown_chars_to)
        # abcdefghijklmnopqrstuvwx (24 chars) are known to the vectorizer,
        # ?!&/() (6 chars) unknown
        # => 6/30 = 1/5 = roughly 20% unknown chars
        texts = create_random_texts(known_chars + unknown_chars,
                                    count_chars,
                                    count_texts)
        matrix = vectorizer.transform(texts, count_chars)
        reverse_transformed = vectorizer.reverse_transform(matrix)
        count_known = 0
        count_unknown = 0
        for text_is in reverse_transformed:
            count_unknown_this = text_is.count(map_unknown_chars_to)
            count_unknown += count_unknown_this
            count_known += len(text_is) - count_unknown_this
        
        count_total = count_known + count_unknown
        fraction_is = float(count_unknown) / float(count_total)
        self.assertTrue(fraction_is > 0.17 and fraction_is < 0.23,
                        "Fraction is %f (%d of %d), expected about 0.2" %
                        (fraction_is, count_unknown, count_total))
    
    def test_vector_lengths(self):
        vectorizer = CharVectorizer("abc", map_unknown_chars_to="1",
                                    fill_left_char="2", fill_right_char="3")
        self.assertEquals(vectorizer.get_one_char_vector_length(),
                          len("abc123"))
        self.assertEquals(vectorizer.get_vector_length(2),
                          len("abc123") * 2)
    
    def test_auto_lower(self):
        vectorizer = CharVectorizer("abcD", map_unknown_chars_to="X",
                                    auto_lowercase=True, auto_uppercase=False)
        texts = ["aaa", "bbb", "ccc", "abc", "AAA", "BBB", "AdD", "EEe", "EeF"]
        expected = ["aaa", "bbb", "ccc", "abc", "aaa", "bbb", "aXD", "XXX", "XXX"]
        matrix = vectorizer.transform(texts, len(texts[0]))
        
        reverse_transformed = vectorizer.reverse_transform(matrix)
        for text_is, text_exp in zip(reverse_transformed, expected):
            self.assertEqual(text_is, text_exp)
    
    def test_auto_upper(self):
        vectorizer = CharVectorizer("ABCd", map_unknown_chars_to="X",
                                    auto_lowercase=False, auto_uppercase=True)
        texts = ["AAA", "BBB", "CCC", "ABC", "aaa", "bbb", "aDd", "eeE", "eEf"]
        expected = ["AAA", "BBB", "CCC", "ABC", "AAA", "BBB", "AXd", "XXX", "XXX"]
        matrix = vectorizer.transform(texts, len(texts[0]))
        
        reverse_transformed = vectorizer.reverse_transform(matrix)
        for text_is, text_exp in zip(reverse_transformed, expected):
            self.assertEqual(text_is, text_exp)
    
    def test_fill_left(self):
        vectorizer = CharVectorizer("abc", fill_left_char="+",
                                    map_unknown_chars_to="X")
        texts = ["a", "aa", "aaa", "b", "bc", "d", "ddd", "abcd"]
        expected = ["++a", "+aa", "aaa", "++b", "+bc", "++X", "XXX", "abc"]
        matrix = vectorizer.transform(texts, 3, fill_right=False)
        
        reverse_transformed = vectorizer.reverse_transform(matrix)
        for text_is, text_exp in zip(reverse_transformed, expected):
            self.assertEqual(text_is, text_exp)
    
    def test_fill_right(self):
        vectorizer = CharVectorizer("abc", fill_right_char="+",
                                    map_unknown_chars_to="X")
        texts = ["a", "aa", "aaa", "b", "bc", "d", "ddd", "abcd"]
        expected = ["a++", "aa+", "aaa", "b++", "bc+", "X++", "XXX", "abc"]
        matrix = vectorizer.transform(texts, 3, fill_right=True)
        
        reverse_transformed = vectorizer.reverse_transform(matrix)
        for text_is, text_exp in zip(reverse_transformed, expected):
            self.assertEqual(text_is, text_exp)
    
    def test_reverse_transform(self):
        vectorizer = CharVectorizer("abc", map_unknown_chars_to="X")
        texts = ["aaa", "bbb", "ccc", "abc", "???", "a?a"]
        expected = ["aaa", "bbb", "ccc", "abc", "XXX", "aXa"]
        matrix = vectorizer.transform(texts, len(texts[0]))
        
        reverse_transformed = vectorizer.reverse_transform(matrix)
        for text_is, text_exp in zip(reverse_transformed, expected):
            self.assertEqual(text_is, text_exp)
    
    def test_reverse_transform_string(self):
        vectorizer = CharVectorizer("abc", map_unknown_chars_to="X")
        texts = ["aaa", "bbb", "ccc", "abc", "???", "a?a"]
        expected = ["aaa", "bbb", "ccc", "abc", "XXX", "aXa"]
        
        matrices = []
        for text in texts:
            matrices.append(vectorizer.transform_string(text, len(texts[0])))
        
        reverse_transformed = []
        for matrix in matrices:
            for row in matrix:
                reverse_transformed.append(vectorizer.reverse_transform_string(row))
        
        for text_is, text_exp in zip(reverse_transformed, expected):
            self.assertEqual(text_is, text_exp)
    
    def test_reverse_transform_char(self):
        vectorizer = CharVectorizer("abc", map_unknown_chars_to="X")
        texts = ["a", "b", "c", "X", "?"]
        expected = ["a", "b", "c", "X", "X"]
        
        matrices = []
        for charr in texts:
            matrices.append(vectorizer.transform_char(charr))
        
        reverse_transformed = []
        for matrix in matrices:
            for row in matrix:
                reverse_transformed.append(
                    vectorizer.reverse_transform_char(row)
                )
        
        for text_is, text_exp in zip(reverse_transformed, expected):
            self.assertEqual(text_is, text_exp)
    
    def test_reverse_transform_maxval(self):
        # test on static example texts
        vectorizer = CharVectorizer("abc", map_unknown_chars_to="X")
        texts = ["aaa", "bbb", "ccc", "abc", "???", "a?a"]
        expected = ["aaa", "bbb", "ccc", "abc", "XXX", "aXa"]
        
        matrix = vectorizer.transform(texts, len(texts[0]))
        rand = np.random.random_sample(matrix.shape)
        matrix = matrix + rand
        
        reverse_transformed = vectorizer.reverse_transform_maxval(matrix)
        for text_is, text_exp in zip(reverse_transformed, expected):
            self.assertEqual(text_is, text_exp)
        
        # test on 1000 random texts
        vectorizer = CharVectorizer(ALPHABET_LOWERCASE,
                                    map_unknown_chars_to="X")
        texts = create_random_texts(ALPHABET_LOWERCASE, 20, 1000)
        expected = texts
        
        matrix = vectorizer.transform(texts, len(texts[0]))
        rand = np.random.random_sample(matrix.shape)
        matrix = matrix + rand
        
        reverse_transformed = vectorizer.reverse_transform_maxval(matrix)
        for text_is, text_exp in zip(reverse_transformed, expected):
            self.assertEqual(text_is, text_exp)
    
    def test_reverse_transform_string_maxval(self):
        vectorizer = CharVectorizer("abc", map_unknown_chars_to="X")
        texts = ["aaa", "bbb", "ccc", "abc", "???", "a?a"]
        expected = ["aaa", "bbb", "ccc", "abc", "XXX", "aXa"]
        
        matrices = []
        for text in texts:
            matrix = vectorizer.transform_string(text, len(texts[0]))
            matrix = matrix + np.random.random_sample(matrix.shape)
            matrices.append(matrix)
        
        reverse_transformed = []
        for matrix in matrices:
            for row in matrix:
                reverse_transformed.append(
                    vectorizer.reverse_transform_string_maxval(row)
                )
        
        for text_is, text_exp in zip(reverse_transformed, expected):
            self.assertEqual(text_is, text_exp)
        
    def test_reverse_transform_char_maxval(self):
        vectorizer = CharVectorizer("abc", map_unknown_chars_to="X")
        texts = ["a", "b", "c", "X", "?"]
        expected = ["a", "b", "c", "X", "X"]
        
        matrices = []
        for charr in texts:
            matrix = vectorizer.transform_char(charr)
            matrix = matrix + np.random.random_sample(matrix.shape)
            matrices.append(matrix)
        
        reverse_transformed = []
        for matrix in matrices:
            for row in matrix:
                reverse_transformed.append(
                    vectorizer.reverse_transform_char_maxval(row)
                )
        
        for text_is, text_exp in zip(reverse_transformed, expected):
            self.assertEqual(text_is, text_exp)

if __name__ == '__main__':
    unittest.main()
