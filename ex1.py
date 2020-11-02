import re
import random
import math


class Ngram_Language_Model:
    FLAVOUR_GENERATE = 1
    FLAVOUR_EVALUATE = 2
    MAX_TRIES = 100000

    def __init__(self, n=3, chars=False):
        self.n = n
        self.chars = chars
        self.model = {}

    def join_array_to_string(self, arr):
        if self.chars:
            return "".join(arr)
        else:
            return " ".join(arr)

    def count_words(self, text):
        """text: string, return count of words"""
        return len(text.split(" "))

    def split_ngram(self, ngram):
        """
        convert ngram string to:
        ngram_suffix: string
        ngram_prefix: array
        """
        ngram_words = self.split_string_to_grams(ngram)
        if len(ngram_words) < self.n:
            return ngram_words, None
        ngram_prefix = ngram_words[-self.n:-1]
        ngram_suffix = ngram_words[-1]
        return ngram_prefix, ngram_suffix

    def get_partial_ngram(self, ngram: str, flavour):
        if ngram is None:
            return None
        """returns ngram to relevant partial according to flavour
        for generates: {a,b,c} => {b,c}
        for evaluate: {a,b,c} => {a,b} """
        ngram_words = self.split_string_to_grams(ngram)
        if flavour == Ngram_Language_Model.FLAVOUR_GENERATE:
            return ngram_words[-self.n + 1:]
        return ngram_words[-self.n:-1]

    def concat(self, current, suffix):
        if self.chars:
            return current + suffix
        return current + ' ' + suffix

    def build_model(self, text):
        model = {}
        grams = self.split_string_to_grams(text)

        if len(grams) < self.n:
            self.model = model
            return

        for i in range(0, len(grams) - self.n + 1):
            sub_list = grams[i: i + self.n]
            if self.chars:
                sub_text = "".join(sub_list)
            else:
                sub_text = " ".join(sub_list)
            if sub_text not in model:
                model[sub_text] = 0
            model[sub_text] = model[sub_text] + 1

        self.model = model

    def get_model(self):
        return self.model

    def split_string_to_grams(self, text):
        """ return an array of grams (words or chars)"""
        if self.chars:
            grams = [char for char in text]
        else:
            grams = text.split(" ")
        return grams

    def get_first_ngram(self, text):
        for ngram, value in self.model.items():
            if ngram.startswith(text):
                return ngram
        return None

    def generate(self, context=None, n=20):
        if context is None:
            result = self.get_ngram_with_prob()
            words_counter = self.n
        else:
            result = context
            words_counter = len(context.split(" "))

        if words_counter < self.n:
            start_ngram = self.get_first_ngram(result)
            if start_ngram is not None:
                result = start_ngram
            else:
                result = self.concat(result, self.get_ngram_with_prob())

        tries = 0
        while tries < Ngram_Language_Model.MAX_TRIES:
            if words_counter < n:
                tries += 1
                new_ngram = self.get_ngram_with_prob(result)
                new_ngram_prefix, new_ngram_suffix = self.split_ngram(new_ngram)
                result = self.concat(result, new_ngram_suffix)
                words_counter = self.count_words(result)
            else:
                return result

        return result

    def build_prob_dict(self, flavour, words=None, ):
        """
        partial_type :
        convert the original model to inverted probability model:
        if words is None - collect all ngrams in the model,
        else, we expect the words to be n-1 length and we collect only relevant ngrams
        original:   { 'who am i' ->2, 'my name is' -> 3}
        inverted: { 2 -> 'who am i', 4-> 'my name is'}
        returns: the new dictionary, the new dictionary range to choose from
        """
        prob_dict = {}
        prob_range = 0
        partial_text = self.get_partial_ngram(words, flavour)
        for ngram, ngram_frequency in self.model.items():
            if words is None:
                prob_dict[prob_range + ngram_frequency] = ngram
                prob_range += ngram_frequency
            else:
                ngram_prefix, ngram_suffix = self.split_ngram(ngram)
                if ngram_prefix == partial_text:
                    prob_dict[prob_range + ngram_frequency] = ngram
                    prob_range += ngram_frequency

        return prob_dict, prob_range

    def get_ngram_with_prob(self, words=None):
        prob_dict, prob_dict_length = self.build_prob_dict(Ngram_Language_Model.FLAVOUR_GENERATE, words)
        rand = random.randint(0, prob_dict_length)
        for prob_limit, ngram_arr in prob_dict.items():
            if rand <= prob_limit:
                return ngram_arr

        # couldn't find anything on the prob_dict, create a new dict, without prefix
        prob_dict, prob_dict_length = self.build_prob_dict(Ngram_Language_Model.FLAVOUR_GENERATE)
        rand = random.randint(0, prob_dict_length)
        for prob_limit, ngram_arr in prob_dict.items():
            if rand <= prob_limit:
                return ngram_arr

    def evaluate(self, text, smooth=False):
        total_probability = 1
        text_ngrams = []
        words = self.split_string_to_grams(text)
        for i in range(0, len(words) - self.n + 1):
            text_ngrams.append(words[i: i + self.n])

        for current_ngram in text_ngrams:
            if smooth:
                total_probability *= self.smooth(self.join_array_to_string(current_ngram))
            else:
                prob = self.evaluate_helper(self.join_array_to_string(current_ngram))
                if prob == 0:
                    return self.evaluate(text, True)
                total_probability *= prob

        return math.log2(total_probability)

    def smooth(self, ngram):
        model_length = len(self.model)
        prob_dict, dict_length = self.build_prob_dict(Ngram_Language_Model.FLAVOUR_EVALUATE, ngram)
        for appearances, dict_ngram in prob_dict.items():
            if ngram == dict_ngram:
                return (1 + appearances) / (dict_length + model_length)

        return 1 / (dict_length + model_length)

    def evaluate_helper(self, ngram):
        prob_dict, dict_range = self.build_prob_dict(Ngram_Language_Model.FLAVOUR_EVALUATE, ngram)
        for appearances, dict_ngram in prob_dict.items():
            if ngram == dict_ngram:
                return appearances / dict_range
        return 0


def normalize_text(text):
    text = re.sub('([.,!?()])', r' \1 ', text.lower())  # add padding on 2 side of the . (or any other punc)
    text = re.sub('\s{2,}', ' ', text)  # merge duplicate padding
    return text.rstrip()  # remove redundant padding in the end


def who_am_i():
    return {'name': 'Ido Gadiel', 'id': '200736494', 'email': 'gadiele@post.bgu.ac.il'}
