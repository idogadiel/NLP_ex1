import re
import random
import math


class Ngram_Language_Model:

    def __init__(self, n=3, chars=False):
        self.n = n
        self.chars = chars
        self.model = {}

    def build_model(self, text):
        model = {}
        grams = self.string_split(text)

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

    def string_split(self, text):
        if self.chars:
            grams = [char for char in text]
        else:
            grams = text.split(" ")
        return grams

    def generate(self, context=None, n=20):
        if context is None:
            result = self.get_ngram_with_prob()
            words_counter = self.n
        else:
            result = context
            words_counter = len(context.split(" "))

        while words_counter < n - 1:
            text_words = self.string_split(result)
            if len(text_words) < self.n:
                text_suffix = text_words
            else:
                text_suffix = text_words[-self.n + 1:]  # check for shorter strings
            new_ngram = self.get_ngram_with_prob(text_suffix)
            ngram_prefix, ngram_suffix = self.split_ngram(new_ngram)
            result += " " + ngram_suffix
            words_counter += 1

        return result

    def build_prob_dict(self, words=None):
        """
        convert the original model to inverted probability model:
        if words is None - collect all ngrams in the model,
        else, we expect the words to be n-1 length and we collect only relevant ngrams
        original:   { 'who am i' ->2, 'my name is' -> 3}
        inverted: { 2 -> 'who am i', 4-> 'my name is'}
        returns: the new dictionary, the new dictionary range to choose from
        """
        prob_dict = {}
        prob_range = 0

        for ngram, ngram_frequency in self.model.items():
            if words is None:
                prob_dict[prob_range + ngram_frequency] = ngram
                prob_range += ngram_frequency
            else:
                ngram_prefix, ngram_suffix = self.split_ngram(ngram)
                if ngram_prefix == words:
                    prob_dict[prob_range + ngram_frequency] = ngram
                    prob_range += ngram_frequency

        return prob_dict, prob_range

    def get_ngram_with_prob(self, words=None):
        # words: array
        prob_dict, prob_dict_length = self.build_prob_dict(words)
        rand = random.randint(0, prob_dict_length)
        for prob_limit, ngram_arr in prob_dict.items():
            if rand <= prob_limit:
                return ngram_arr

        prob_dict, prob_dict_length = self.build_prob_dict()
        for prob_limit, ngram_arr in prob_dict.items():
            if rand <= prob_limit:
                return ngram_arr

    def split_ngram(self, ngram):
        """
        convert ngram string to:
        ngram_suffix: string
        ngram_prefix: array
        """
        ngram_words = self.string_split(ngram)
        ngram_prefix = ngram_words[0:self.n - 1]
        ngram_suffix = ngram_words[-1]
        return ngram_prefix, ngram_suffix

    def evaluate(self, text, smooth=False):
        total_probability = 1
        text_ngrams = []
        words = text.split()
        for i in range(0, len(words) - self.n + 1):
            text_ngrams.append(words[i: i + self.n])

        for current_ngram in text_ngrams:
            if smooth:
                total_probability *= self.smooth(current_ngram)
            else:
                prob = self.evaluate_helper(current_ngram)
                if prob == 0:
                    return self.evaluate(text, True)
                total_probability *= prob

        return math.log2(total_probability)

    def smooth(self, ngram):
        current_ngram_prefix = ngram[0:self.n - 1]
        model_length = len(self.model)
        prob_dict, dict_length = self.build_prob_dict(current_ngram_prefix)
        for appearances, dict_ngram in prob_dict.items():
            if ngram == dict_ngram.split():
                return (1 + appearances) / (dict_length + model_length)
        return 1 / (dict_length + model_length)

    def evaluate_helper(self, ngram):
        current_ngram_prefix = ngram[0:self.n - 1]
        prob_dict, dict_range = self.build_prob_dict(current_ngram_prefix)
        for appearances, dict_ngram in prob_dict.items():
            if ngram == dict_ngram.split():
                return appearances / dict_range
        return 0


def normalize_text(text):
    text = re.sub('([.,!?()])', r' \1 ', text.lower())  # add padding on 2 side of the . (or any other punc)
    text = re.sub('\s{2,}', ' ', text)  # merge duplicate padding
    return text.rstrip()  # remove redundant padding in the end


def who_am_i():
    return {'name': 'Ido Gadiel', 'id': '200736494', 'email': 'gadiele@post.bgu.ac.il'}
