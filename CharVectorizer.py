# -*- coding: utf-8 -*-
"""A class to turn a list of strings into concatenated one hot vectors of
each char."""
import re
import numpy as np

# Uncomment this line and the respective code in the method _one_hot_matrix()
# to use scikit's OneHotEncoder for the conversion into one hot vectors.
# There seems to be no performance difference between that vectorizer and
# the current numpy code.
#from sklearn.preprocessing import OneHotEncoder

class CharVectorizer(object):
    """A class to convert lists of strings into concatenated one hot vectors
    (one one-hot-vector per char).

    == Example, Introduction ==

    Example usage for two string windows:
        vectorizer = CharVectorizer("abcdefghijklmnopqrstuvwxyz")
        windows = [
            "the fox jumped over the fence.",
            "another fox did something. And"
        ]
        target_length = max(len(window) for window in windows)
        matrix = vectorizer.transform(windows, target_length)

    The resulting matrix is a numpy matrix.
    You may feed it e.g. into a neural network.
    Each row of the matrix is a long concatenated vector made up of
    one hot vectors of each char. See more for that in transform().


    == accepted_chars and map_unknown_chars_to ==

    The class expects the chars to project onto one hot vectors (given
    in accepted_chars) and another char (map_unknown_chars_to)
    which will be used as a mapping for chars that are not contained
    in accepted_chars. Notice that map_unknown_chars_to itself may be
    contained in accepted_chars. E.g. you may set
        accepted_chars = "ABCX"
    and
        map_unknown_chars_to = "X".
    Then there will be one hot vectors for A, B, C and X. Any other
    char will get the one hot vector of X. (Therefore, X will be
    ambiguous.) You do not have to add map_unknown_chars_to to
    accepted_chars, it will be added automatically if neccessary.


    == target length, fill_left_char, fill_right_char ==

    Notice that in the example above, the target_length gets calculated
    dynamically. In most cases however, you would know your intended
    window length (e.g. 50 chars) and should use that number.
    You can safely provide windows that are longer or shorter than
    that window length, they will automatically be increased/decreased
    in size.
    If one of the strings must be increased in length that will either
    happen by adding chars on the left of that string (at its start)
    or by appending chars at the right of that string (at its end).
    Which of option will be chosen is defined by the parameter
    fill_right in transform(). If chars are appended on the right, then
    fill_right_char will be used for that, otherwise fill_left_char.


    == auto_lowercase and auto_uppercase ==

    This vectorizer can automatically project chars onto their lowercase
    or uppercase variants, when neccessary. E.g. set auto_lowercase to
    True in order to automatically lowercase chars. Notice that lowercase
    will only be used, if the uppercase variant isn't contained in
    accepted_chars (or in map_unknown_chars_to, fill_left_char and
    fill_right_char). The lowercase letter must also be in one of the
    mentioned parameters.


    == Reversing transformations ==

    You can reverse the transformation by using reverse_transform(matrix),
    e.g.
        ...
        matrix = vectorizer.transform(windows, target_length)
        strings = vectorizer.reverse_transform(matrix)
    If you have a kinda fuzzy matrix (e.g. the result of a neural network
    output) then you may use reverse_transform_maxval() instead. It choses
    the maximum value per one hot vector as 1 and all others as 0.

    Attributes:
        auto_lowercase: Whether to automatically map uppercase chars to
            lowercase ones if neccessary and possible.
        auto_uppercase: Whether to automatically map lowecase chars to
            uppercase ones if neccessary and possible.
        map_unknown_chars_to: A char onto which all chars will be mapped
            that are not explicitly contained in accepted_chars.
        fill_left_char: A character to use, when filling up too short
            strings on the left. Will be added internally to accepted_chars
            automatically.
        fill_right_char: A character to use, when filling up too short
            strings on the right. Will be added internally to accepted_chars
            automatically.
        accepted_chars: A list of chars that are accepted by this vectorizer.
            That means that they will be mapped to one hot vectors instead
            of using the one hot vector of the char map_unknown_chars_to.
            Notice that this is not identical to the parameter accepted_chars
            in the constructor, as this list contains also the chars
            map_unknown_chars_to, fill_left_char and fill_right_char.
        accepted_chars_indirect: The same list as accepted_chars, extended
            by lowercase and uppercase variants of chars (if auto_lowercase
            and/or auto_uppercase is set to True).
        char_to_index: A dictionary mapping each char to a unique index.
            That index is rougly the same as the position in accepted_chars.
            (With some additional chars in the case of
            auto_lowercase/auto_uppercase.)
        unknown_chars_regex: A regex that matches all chars that are not
            contained in accepted_chars_indirect."""

    def __init__(self, accepted_chars, map_unknown_chars_to='#',
                 fill_left_char=' ', fill_right_char=' ',
                 auto_lowercase=True, auto_uppercase=False):
        """Creates a new instance of the CharVectorizer.


        == Example, Introduction ==

        Example usage for two string windows:
            vectorizer = CharVectorizer("abcdefghijklmnopqrstuvwxyz")
            windows = [
                "the fox jumped over the fence.",
                "another fox did something. And"]
            target_length = max(len(window) for window in windows)
            matrix = vectorizer.transform(windows, target_length)

        The resulting matrix is a numpy matrix.
        You may feed it e.g. into a neural network.
        Each row of the matrix is a long concatenated vector made up of
        one hot vectors of each char. See more for that in transform().


        == accepted_chars and map_unknown_chars_to ==

        The class expects the chars to project onto one hot vectors (given
        in accepted_chars) and another char (map_unknown_chars_to)
        which will be used as a mapping for chars that are not contained
        in accepted_chars. Notice that map_unknown_chars_to itself may be
        contained in accepted_chars. E.g. you may set
            accepted_chars = "ABCX"
        and
            map_unknown_chars_to = "X".
        Then there will be one hot vectors for A, B, C and X. Any other
        char will get the one hot vector of X. (Therefore, X will be
        ambiguous.) You do not have to add map_unknown_chars_to to
        accepted_chars, it will be added automatically if neccessary.


        == target length, fill_left_char, fill_right_char ==

        Notice that in the example above, the target_length gets calculated
        dynamically. In most cases however, you would know your intended
        window length (e.g. 50 chars) and should use that number.
        You can safely provide windows that are longer or shorter than
        that window length, they will automatically be increased/decreased
        in size.
        If one of the strings must be increased in length that will either
        happen by adding chars on the left of that string (at its start)
        or by appending chars at the right of that string (at its end).
        Which of option will be chosen is defined by the parameter
        fill_right in transform(). If chars are appended on the right, then
        fill_right_char will be used for that, otherwise fill_left_char.


        == auto_lowercase and auto_uppercase ==

        This vectorizer can automatically project chars onto their lowercase
        or uppercase variants, when neccessary. E.g. set auto_lowercase to
        True in order to automatically lowercase chars. Notice that lowercase
        will only be used, if the uppercase variant isn't contained in
        accepted_chars (or in map_unknown_chars_to, fill_left_char and
        fill_right_char). The lowercase letter must also be in one of the
        mentioned parameters.


        == Reversing transformations ==

        You can reverse the transformation by using reverse_transform(matrix),
        e.g.
            ...
            matrix = vectorizer.transform(windows, target_length)
            strings = vectorizer.reverse_transform(matrix)
        If you have a kinda fuzzy matrix (e.g. the result of a neural network
        output) then you may use reverse_transform_maxval() instead. It choses
        the maximum value per one hot vector as 1 and all others as 0.


        Args:
            accepted_chars: List of characters that are ought to be transformed
                to one hot vectors.
            map_unknown_chars_to: A character onto which all characters will
                be mapped on that are not contained in accepted_chars. Will
                be internally added to accepted_chars automatically.
                (Default: "#")
            fill_left_char: A character to use, when filling up too short
                strings on the left. Will be added internally to accepted_chars
                automatically. (Default: " ")
            fill_right_char: A character to use, when filling up too short
                strings on the right. Will be added internally to accepted_chars
                automatically. (Default: " ")
            auto_lowercase: Set to True to allow mapping of uppercase chars to
                lowercase ones. All characters that are uppercase and not
                internally contained in accepted_chars will then be mapped
                to their lowercase variant, so long as the lowercase variant
                is contained in accepted_chars. (So: map only if neccessary
                and possible.) (Default: True)
            auto_uppercase: Set to True to allow mapping of lowercase chars to
                uppercase ones. (See auto_lowercase.) (Default: False)"""
        self.auto_lowercase = auto_lowercase
        self.auto_uppercase = auto_uppercase
        self.map_unknown_chars_to = map_unknown_chars_to
        self.fill_left_char = fill_left_char
        self.fill_right_char = fill_right_char

        # We create two lists of accepted chars here:
        #  1. accepted_chars: The accepted chars as provided in the
        #     parameter and additionaly the chars
        #     map_unknown_chars_to, fill_left_char, fill_right_char,
        #     which are indirectly always accepted.
        #     Notice that we keep their order here via
        #      unique_keep_order, so that the results of this vectorizer
        #      are always the same if the same parameters are provided.
        #  2. accepted_chars_indirect: Same as accepted_chars, but on
        #     top of that this list contains additional chars if
        #     auto_lowercase is True (then all chars c which are uppercase
        #     (c == c.upper()) and which are not in self.accepted_chars,
        #     but which's lowercase variant (c.lower()) is in
        #     self.accepted_chars. Analogous for lowercase chars if
        #     auto_uppercase is True.
        self.accepted_chars = self._unique_keep_order(
            list(accepted_chars) + [
                map_unknown_chars_to, fill_left_char, fill_right_char
            ]
        )
        # notice: this list will be extended in the next for loop
        self.accepted_chars_indirect = list(self.accepted_chars)

        # Create a dictionary that maps chars to their index
        # in the list self.accepted_chars.
        # Additionally, if auto_lowercase is True then all chars that
        # are uppercase variants of a char in self.accepted_chars will
        # be mapped to their lowercase index, unless their uppercase
        # variant is already contained in self.accepted_chars.
        # (Same goes for auto_uppercase=True mapping lowercase chars
        # to uppercase ones.)
        #
        # Notice that we do not map uppercase chars to their lowercase
        # ones if their lowercase char is not contained in
        # self.accepted_chars. (They will be mapped to
        # map_unknown_chars_to later on.)
        #
        # Notice also that this dictionary does not contain every
        # existing chars, just the ones that can be mapped according
        # to self.accepted_chars.
        #
        # We will use this dictionary later on to map every char in
        # all given strings to an index.
        self.char_to_index = dict()
        for i, charr in enumerate(self.accepted_chars):
            self.char_to_index[charr] = i

            if auto_lowercase:
                charr_upp = charr.upper()
                if charr_upp != charr and charr_upp not in self.accepted_chars:
                    self.accepted_chars_indirect.append(charr_upp)
                    self.char_to_index[charr_upp] = i

            if auto_uppercase:
                charr_low = charr.lower()
                if charr_low != charr and charr_low not in self.accepted_chars:
                    self.accepted_chars_indirect.append(charr_low)
                    self.char_to_index[charr_low] = i

        # A regex that matches all characters that are not in
        # self.accepted_chars_indirect.
        # We will use this regex later on to find chars that
        # cannot be mapped to an index in self.accepted_chars and
        # replace them by the index of the char given via
        # map_unkown_chars_to.
        self.unknown_chars_regex = re.compile(
            r"[^%s]" % re.escape("".join(self.accepted_chars_indirect))
        )

    def fit(self, _):
        """Fit vectorizer to dataset.

        This method only exists to keep the interface comparable to
        scikit-learn.

        Args:
            _: A list of strings (ignored).
        Returns:
            self"""
        return self

    def fit_transform(self, texts, per_string_length, fill_right=True,
                      dtype=np.int):
        """This method is completely identical to transform().

        This method only exists to keep the interface comparable to
        scikit-learn. You can ignore the fit() part (see fit()).

        Args:
            see transform()
        Returns:
            see transform()
        """
        return self.transform(texts, per_string_length,
                              fill_right=fill_right, dtype=dtype)

    def transform(self, texts, per_string_length, fill_right=True,
                  dtype=np.int):
        """Transform a list of strings in a char-by-char way into a matrix of
        concatenated one hot vectors.

        This method receives a list of strings (in texts), their intended
        lengths (per_string_length) and optionally whether too short strings
        should be filled up on the right (at their end) or on the left (at
        their start).
        The method will convert every single char into a one hot vector,
        e.g. "a" might be turned into [1, 0, 0], "b" into [0, 1, 0] and
        "c" into [0, 0, 0]. The method will then concatenate for each of the
        strings all of their one hot vectors to one big vector,
        e.g. "abc" might be turned into [1,0,0, 0,1,0, 0,0,0]. This is done
        for every string, so the result will be one big matrix, where each
        row resembles a string and the columns are made up by one hot
        vectors.

        Every row in the matrix must have the same length, therefore that
        length must be provided by per_string_length.
        Strings that are too short for that length will automatically be
        increased in length. By default that happens by appending the character
        self.fill_right_char (see constructor) to the end as often as
        necessary. If fill_right is set to False, the character
        self.fill_left_char will instead be added to the start/left of
        the string (as often as necessary).
        Too long strings will simply be cut to the required length.

        You can savely call this without calling fit() before.

        If you want to know the length of each row of the resulting matrix
        (before generating one) you can call
        get_vector_length(per_string_length).

        Args:
            texts: A list of strings, e.g. ["dog", "cat", "foo"].
            per_string_length: The target length of each of the provided
                strings. Too long strings will be shortened, too short ones
                will be filled up with chars in
                self.fill_left_char/self.fill_right_char.
            fill_right: Whether to fill up too short strings on their
                right (True) or their left (False). (Default: True)
            dtype: Datatype of the returned numpy matrix. (Default: np.int)

        Returns:
            Two dimensional numpy matrix,
            where each row contains concatenated one hot vectors that
            model the string at the same position in the input list."""

        rows = len(texts)
        cols = per_string_length
        matrix = np.zeros((rows, cols), dtype=np.int)

        for i, text in enumerate(texts):
            matrix[i, :] = self._text_to_char_indexes(
                text,
                per_string_length=per_string_length,
                fill_right=fill_right
            )

        return self._one_hot_matrix(matrix, per_string_length,
                                    len(self.accepted_chars) - 1,
                                    dtype=dtype)

    def transform_string(self, text, per_string_length, fill_right=True,
                         dtype=np.int):
        """Transforms a single string into a long vector of concatenated
        one hot vectors.

        See transform().

        Notice: Calling this method many times is probably slow. Use a list
        of strings for transform() instead.

        Args:
            text: The string to transform.
            per_string_length: The target length of the provided
                string. Too long strings will be shortened, too short ones
                will be filled up with chars in
                self.fill_left_char/self.fill_right_char.
            fill_right: Whether to fill up too short strings on their
                right (True) or their left (False). (Default: True)
            dtype: Datatype of the returned numpy matrix. (Default: np.int)

        Returns:
            Two dimensional numpy matrix with a single row containing
            the concatenated one hot vectors of each char in the string.
        """
        return self.transform([text],
                              per_string_length,
                              fill_right=fill_right,
                              dtype=dtype)

    def transform_char(self, char, dtype=np.int):
        """Transforms a single char into a one hot vector.

        This method simply treats the char like a string of length one
        and therefor calls transform_string().

        Notice: Calling this method many times is probably slow.
        It's better to convert your chars to a list of chars. Then just view
        it like a list of strings and call transform(), e.g. something like:
            transform(list("a fox jumped over the fence"), 1)

        Args:
            char: The char to transform.
            dtype: Datatype of the returned numpy matrix. (Default: np.int)

        Returns:
            Two dimensional numpy matrix with a single row containing
            a one hot vector for your char."""
        return self.transform_string(char,
                                     1,
                                     dtype=dtype)

    def _text_to_char_indexes(self, text, per_string_length,
                              fill_right=True):
        """Transforms a string into a list of character indices.

        The indices roughly match the position in self.accepted_chars.
        (See more in __init__().)

        Args:
            text: The string to transform.
            per_string_length: The target length of the string. If the
                provided string (text) is longer, it will automatically be
                shortened. If it is shorter, it will be filled up with
                self.fill_left_char on the left or self.fill_right_char
                on the right.
            fill_right: If True, too short strings will be filled up
                with self.fill_right_char on the right until
                per_string_length is reached. If it is false, the same will
                be done on the left with self.fill_left_char.

        Returns:
            A list of integer indices, roughly matching be index of each char
            in self.accepted_chars.
            E.g. "abcb" might be turned into [0, 1, 2, 1]."""
        text = self.unknown_chars_regex.sub(self.map_unknown_chars_to, text)

        # notice: using a numpy array here seems to be a bit faster
        # than using a normal list()
        result = np.zeros((per_string_length,), dtype=np.int)

        # Limit the given string (text) to the given target string
        # length (per_string_length)
        lenn = len(text)
        if lenn > per_string_length:
            # The string (text) is too long, shorten it to the target
            # length
            text = text[0:per_string_length]
        elif lenn < per_string_length:
            # The string (text) is too short. Increase the length
            # either by filling it up on the left side with
            # the char in self.fill_left_char or alternatively
            # by doing the same on the right side
            # with self.fill_right_char.
            diff = per_string_length - lenn
            if fill_right:
                filler = self.fill_right_char * diff
                text = text + filler
            else:
                filler = self.fill_left_char * diff
                text = filler + text

        # Now we are able to map every char in the given string (text)
        # to an index by using the dictionary self.char_to_index,
        # which was defined in the constructor.
        # Every char always gets the same index, e.g. 'a' might be
        # mapped to 5 and '?' to 37.
        for i, charr in enumerate(list(text)):
            index = self.char_to_index[charr]
            result[i] = index

        return result

    def reverse_transform(self, matrix):
        """Takes the matrix of concatenated one hot vectors of multiple strings
        and returns these original strings.

        Notice that this method is not optimized for performance.

        Args:
            matrix: The matrix of vectors, where each row is a string's vector.
                (So the result of transform().)

        Returns:
            List of strings."""
        assert type(matrix).__module__ == np.__name__

        result = []
        for row in matrix:
            result.append(self.reverse_transform_string(row))
        return result

    def reverse_transform_maxval(self, matrix):
        """Takes a matrix of concatenated fuzzy one hot vectors (e.g. neural
        network output) of multiple strings and tries to return the
        best matching strings.

        See reverse_transform_string_maxval() for more.

        Notice that this method is not optimized for performance.

        Args:
            matrix: The matrix of vectors, where each row is a string's vector.
                (So something similar to the result of transform().)

        Returns:
            List of strings."""
        assert type(matrix).__module__ == np.__name__

        result = []
        for row in matrix:
            result.append(self.reverse_transform_string_maxval(row))
        return result

    def reverse_transform_string(self, vectorized):
        """Takes the concatenated one hot vectors of multiple chars and
        returns these chars.

        Notice that this method is not optimized for performance.

        Args:
            vectorized: The concatenated one hot vectors of multiple chars
                (one single string, hence the "_string" in the method name).
                This is identical to one row in the resulting matrix after
                calling fit_transform()/transform().

        Returns:
            The result are the decoded chars (so a string).

            Notice that the reverse transformation may be lossful, i.e. some
            chars may get replaced by a placeholder (see the parameter
            "map_unknown_chars_to" in __init__()) or by their
            lowercase/uppercase variants."""
        assert type(vectorized).__module__ == np.__name__

        # Convert numpy array to list
        #vectorized = list(vectorized)

        # How long is the one hot vector of a single char?
        length_per_char = self.get_one_char_vector_length()

        # The vector "vectorized" contains n chars
        # as (concatenated) one hot vectors, each having
        # length length_per_char.
        # We will now split "vectorized" into n of these vectors,
        # so that we can then map every one hot vector to the
        # respective char.
        vecs = self._list_to_chunks(vectorized, length_per_char)

        # Transform every single one hot vector into the respective
        # char. The result is a list of chars.
        # Notice that this is lossful. Chars that were replaced
        # by the char self.map_unknown_chars_to will not be recoverable.
        # Chars that were automatically replaced by their lowercase
        # variants will also stay lowercased.
        text = [self.reverse_transform_char(vec) for vec in vecs]

        # Convert the list of chars to one string.
        return "".join(text)

    def reverse_transform_string_maxval(self, vectorized):
        """Takes the fuzzy versions of concatenated one hot vectors of multiple
        chars and tries to reconstruct these chars.

        This method is mostly identical to reverse_transform_string().
        In contrast to reverse_transform_string() this method does not expect the
        input to be purely made of 0's and 1's. Instead it will simply
        interpret the maximum value in each single one hot vector as the 1 and
        all other components as 0's. So an input [0.7, 0.3] could be
        interpreted as [1, 0] by this method (assuming that it contains just
        a single one hot vector).

        This method is intended for neural network outputs, which may not
        contain perfect one hot vectors.

        Notice that this method is not optimized for performance.

        Args:
            vectorized: The concatenated one hot vectors of multiple chars
                of a single string (hence the "_string" in the method name),
                e.g. the output of a neural network.
                This is similar to one row in the resulting matrix after
                calling fit_transform()/transform().
                Of each one hot vector only the maximum value will be viewed
                as a one, so the input may in fact not contain a single one.

        Returns:
            The result are the decoded chars (so a string).

            Notice that the reverse transformation may be lossful, i.e. some
            chars may get replaced by a placeholder (see the parameter
            "map_unknown_chars_to" in __init__()) or by their
            lowercase/uppercase variants.

            (Identical result to reverse_transform_string().)
        """
        assert type(vectorized).__module__ == np.__name__

        # Convert numpy array to list.
        #vectorized = list(vectorized)

        # See reverse_transform()
        length_per_char = self.get_one_char_vector_length()

        # See reverse_transform()
        vecs = self._list_to_chunks(vectorized, length_per_char)

        # We now have a list of vectors that are not yet one hot
        # vectors. We will search in each one of them for the
        # component that has the highest value and set it to 1.
        # The other components will be set to 0.
        text = []
        for vec in vecs:
            text.append(self.reverse_transform_char_maxval(vec))

        # See reverse_transform()
        return "".join(text)

    def reverse_transform_char(self, char_one_hot_vector):
        """Takes a single char's one hot vector and returns the corresponding
        char.

        Notice that this method is not optimized for performance.

        Args:
            char_one_hot_vector: The char's one hot vector.

        Returns:
            A char.

            It will result in an assertion error if there is no 1 in the
            vector. If there are multiple 1's, the first position
            will be picked and interpreted as the only 1 in the vector.
            There will also be an assertion error if the provided vector is
            empty or longer than it should be."""
        # This now simply maps to reverse_transform_char_maxval()
        # alternative would be:
        #   idx = np.where(char_one_hot_vector == 1)[0][0]
        #   return self.accepted_chars[idx]
        return self.reverse_transform_char_maxval(char_one_hot_vector)

    def reverse_transform_char_maxval(self, fuzzy_one_hot_vector):
        """Takes a single char's fuzzy one hot vector (e.g. neural network
        output) and tries to return the best matching char.

        See reverse_transform_string_maxval() for more.

        Notice that this method is not optimized for performance.

        Args:
            fuzzy_one_hot_vector: The char's fuzzy one hot vector.

        Returns:
            A char.

            An assertion error will be raised if the provided vector is
            empty (length is zero) or the vector is longer than it should be."""
        assert type(fuzzy_one_hot_vector).__module__ == np.__name__
        assert fuzzy_one_hot_vector.shape == (len(self.accepted_chars),)

        # Find the component with the highest value.
        #max_index = fuzzy_one_hot_vector.index(max(fuzzy_one_hot_vector))
        #print fuzzy_one_hot_vector
        #max_index = np.where(fuzzy_one_hot_vector == 1)[0][0]
        max_index = np.argmax(fuzzy_one_hot_vector)

        # Return the char at the position of the highest component.
        return self.accepted_chars[max_index]

    def get_one_char_vector_length(self):
        """Get the length of each char's one hot vector.

        This is similar, but not always identical to the length of the parameter
        accepted_chars in __init__. See __init__ for more info.

        Returns:
            Length of each char's one hot vector (integer)."""
        return len(self.accepted_chars)

    def get_vector_length(self, per_string_length):
        """Get the length of a vectorized string.

        This is identical to the length of each row in the resulting
        one hot matrix.
        It's also identical to
            (length of each character's one hot vector) * (per_sting_length).
        You will have to provide the target length of the string as this
        vectorizer only accepts equaly-sized strings and will prune
        too long ones or fill up too short ones.
        Args:
            per_string_length: Target length of each complete string of the
                input data. E.g. if your string windows have target lengths
                of 50 characters each, then chose 50.
        Returns:
            An integer representing the length of each string's full
            vector (containing many conatenated one hot character vectors)."""
        return per_string_length * self.get_one_char_vector_length()

    def _list_to_chunks(self, lst, chunk_length):
        """A method to split a list into smaller sublists of specified
        length.

        Args:
            lst: The list to split.
            chunk_length: Length of each sublist.
        Returns:
            Smaller sized sublists, e.g. [1,2,3,4] with chunk_length=2 would
            be split into [1,2], [3,4]. The last list might not be of
            length chunk_length but smaller
            (precisely when len(lst) % chunk_length != 0)."""
        for i in xrange(0, len(lst), chunk_length):
            yield lst[i:i + chunk_length]

    def _unique_keep_order(self, seq):
        """Removes duplicate entries from a list while not changing the order
        of any remaining entry.

        Args:
            seq: The list to change.
        Returns:
            The same list with the same ordering, but without duplicate entries.
        """
        seen = set()
        seen_add = seen.add
        return [x for x in seq if not (x in seen or seen_add(x))]

    def _one_hot_matrix(self, char_indexes, per_string_length, max_char_index,
                        dtype=np.int):
        """Transform a list of lists of integer-features (matrix of features)
        into concatenated one hot representations of these features.

        E.g. if [[1,2],[2,2]] is provided, it will (for the first row)
        convert 1 into a one hot representation v_1 (e.g. [1,0])
        and then 2 into a one hot representation v_2 (e.g. [0,1]).
        It will then create a concatenated list v_1 + v_2 of the two (e.g.
        [1,0,0,1]). That list will be the first row. It will do the same for
        the second row (e.g. [0,1,0,1], so the result would
        be [[1,0,0,1],[0,1,0,1]]).

        Args:
            char_indexes: The list of lists of features (= matrix of features).
                Every cell must contain an integer. E.g. the strings
                ["ab", "bb"] could be represented by [[0, 1], [1, 1]].
            per_string_length:
            max_char_index: Maximum possible index to represent a char
                in the matrix. (The value does not have to be used in the
                matrix.) This value plus 1 equals the size of each one hot
                vector.
            dtype: Numpy dtype of the resulting matrix.

        Returns:
            A large matrix in which every feature of every row is replaced by
            a one hot vector. The one hot vectors of each row are concatenated
            to one large vector.
            E.g. [[0, 1], [1, 1]] could result in
            [[1,0,0,1],[0,1,0,1]] or split visually into one hot vectors:
              1,0|0,1   0,1|0,1.
        """
        # Uncomment this to use scikit's OneHotEncoder.
        # Performance Test:
        # ~0.022s for 1000 examples, each 100 chars
        #enc = OneHotEncoder(n_values=max_char_index+1, sparse=False, dtype=np.int)
        #return enc.fit_transform(char_indexes)

        # Uncommented this to create one hot vectors using numpy.eye()
        # with numpy's ravel().
        # Performance Test:
        # ~0.08s for 1000 Examples, each 100 chars
        #arr = np.eye(max_char_index + 1)[char_indexes.ravel()]
        #(rows, cols) = (char_indexes.shape[0], per_string_length * (max_char_index + 1))
        #arr = np.reshape(arr, (rows, cols))
        #return arr

        # The following code creates one large matrix filled with 0's
        # and then sets the necessary indices to 1 using
        # numpy.repeat() and numpy.cumsum() to find these indices and
        # then arr.flat[indices]=1 to set them to 1.
        # The performance is similar to OneHotEncoder. This method was
        # chosen as it doesn't depend on scikit.
        #
        # Performance Test:
        # Using arr.flat access with cumulative offset
        # ~0.022s for 1000 examples, each 100 chars
        n_rows = char_indexes.shape[0]
        n_features = per_string_length
        n_values_per_feature = max_char_index + 1

        # Create a matrix of the necessary shape full of zeros.
        arr = np.zeros((n_rows, n_features * n_values_per_feature),
                       dtype=dtype)

        # We cannot just use the indices of char_indexes as these are
        # the indices of chars in self.accepted_chars. Using only them
        # would only set the first N cells in the matrix again at again
        # to 1 (for N chars on self.accepted_chars).
        # We basically need offsets to e.g. know that we are operating
        # on the i'th one hot vector (representing the i'th char in
        # all strings) and then use the indices to set the j'th component
        # of that one hot vector to 1.
        # So now we build the offsets, which have to match the one hot
        # vector lenghts. If a one hot vector has length N then we first
        # create a array of the form [N, N, N, N, ...] with M times N,
        # where M is the total amount of chars in all of our strings.
        # We then set the first component to zero and
        # get [0, N, N, N, ...]. Then we cumulate the values to
        # get [0, 0+N, 0+N+N, 0+N+N+N, ...] = [0, 1N, 2N, 3N, ...]. And
        # those values are exactly the required offsets.
        offset = np.repeat([n_values_per_feature], n_rows * per_string_length)
        offset[0] = 0
        offset = np.cumsum(offset)

        # Set the the "correct" component of each one hot vector to 1.
        # See descriptions above for more info.
        # Notice that flat[] allows indixing into the matrix as if it
        # was a 1D-Array.
        # Notice that ravel() converts our matrix also to 1D form.
        # (And the offsets are 1D too.)
        arr.flat[offset + char_indexes.ravel()] = 1

        return arr
