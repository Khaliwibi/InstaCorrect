#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 01 15:36:48 2017
"""

import random
import string
from random import randint
from nltk.tokenize import word_tokenize
import numpy as np
import nltk
from nltk.tokenize.moses import MosesDetokenizer

class Mistake(object):
    """
    Give it a sentence, it will mess it up and return lots of sentence with mistakes in them
    """

    def __init__(self):
        self.methods = [method for method in dir(Mistake) if method.startswith('_m_')]
        self.detokenizer = MosesDetokenizer()
        self.nb_methods = len(self.methods)
        
    def generate_mistake(self, sentence):
        self.word_tokenize = word_tokenize(sentence)
        self.rg = list(range(len(sentence)))
        number_of_mistakes = (len(sentence) // 100) + 1
        counter = 0
        while (counter < number_of_mistakes) and (counter < 100):
            method = randint(0, self.nb_methods-1)
            continue_or_not, sentence = getattr(self, self.methods[method])(sentence)
            if continue_or_not:
                counter +=1
        return sentence

    def generate_mistakes(self, sentence):
        mistakes = []
        number_of_mistakes = (len(sentence) // 75)
        for index, method in enumerate(self.methods):
            randoms = [index] + np.random.choice(len(self.methods)-1, number_of_mistakes).tolist()
            mistake = sentence
            for num in randoms:
                mistake = getattr(self, self.methods[num])(mistake)
            mistakes.append(mistake)
        return mistakes

    def one_by_the_other(self, sentence, one, two):
        if sentence.find(one) > 0:
            return True, sentence.replace(one, two, 1)
        elif sentence.find(two) > 0:
            return True, sentence.replace(two, one, 1)
        else:
            return False, sentence

    def replace_end_word(self, sentence_str, one, two, howmany): #replace_end_word(sentence_str, "s", "", 1)
        sentence = word_tokenize(sentence_str)
        rg = list(range(len(sentence)))
        random.shuffle(rg)
        found = False
        for i in rg:
            if sentence[i].endswith(one) > 0:
                sentence[i] = sentence[i][:-howmany]+two
                found = True
        return found, self.detokenizer.detokenize(sentence, return_str=True)

    def _insertChar(self, mystring, position, chartoinsert):
        """
        Randomly insert a character into a sentence. Mimik a spelling mistake.
        """
        mystring = mystring[:position] + chartoinsert + mystring[position:]
        return mystring

    def _choose_random(self, sentence, choices):
        i = randint(0,len(choices)-2)
        for ele in choices:
            new_list = list(choices)
            new_list.remove(ele)
            if sentence.find(ele) > 0:
                return True, sentence.replace(ele, new_list[i], 1)
        return False, sentence

    def _m_add_random_mistake(self, sentence, num_mistakes=randint(1,5)):
        for _ in range(num_mistakes):
            sentence = self._insertChar(sentence, randint(0,len(sentence)), random.choice(string.ascii_letters).lower())
        return True, sentence

    def _m_er_mistake(self, sentence_str):
        return self.replace_end_word(sentence_str, "er", u"é", 2)

    def _m_e_mistake(self, sentence_str):
        return self.replace_end_word(sentence_str, "e", "", 1)

    def _m_e_mistake_bis(self, sentence_str):
        sentence = word_tokenize(sentence_str)
        i = randint(0,len(sentence)-1)
        sentence[i] = sentence[i] + 'e'
        return True, self.detokenizer.detokenize(sentence, return_str=True)

    def _m_s_mistake(self, sentence_str):
        return self.replace_end_word(sentence_str, "s", "", 1)

    def _m_s_mistake_bis(self, sentence_str):
        sentence = word_tokenize(sentence_str)
        i = randint(0,len(sentence)-1)
        sentence[i] = sentence[i] + 's'
        return True, self.detokenizer.detokenize(sentence, return_str=True)

    def _m_remove_union(self, sentence_str):
        if sentence_str.find('-') > 0:
            return True, sentence_str.replace('-', '', 1)
        else:
            return False, sentence_str

    def _m_switch_cent_to_cents_vice(self, sentence_str):
        return self.one_by_the_other(sentence_str, 'cent', 'cents')

    def _m_switch_mille_to_milles_vice(self, sentence_str):
        return self.one_by_the_other(sentence_str, 'mille', 'milles')

    def _m_switch_un_to_une_vice(self, sentence_str):
        return self.one_by_the_other(sentence_str, 'un', 'une')

    def _m_switch_nn_to_n_vice(self, sentence_str):
        return self.one_by_the_other(sentence_str, 'nn', 'n')

    def _m_switch_quel(self, sentence_str):
        return self._choose_random(sentence_str, ['laquelle', 'lequel', 'auxquelles', 'auquel', 'lesquels'])

    def _m_switch_a(self, sentence_str):
        return self.one_by_the_other(sentence_str, u' à ', ' a ')

    def _m_switch_ll(self, sentence_str):
        return self.one_by_the_other(sentence_str, 'll', 'l')

    def _m_switch_tt(self, sentence_str):
        return self.one_by_the_other(sentence_str, 'tt', 't')

    def _m_replace_accents(self, sentence_str):
        return self.one_by_the_other(sentence_str, u'é', u'è')

    def _m_pronoms_mon(self, sentence_str):
        return self._choose_random(sentence_str, ['mon', 'ma', 'mes'])

    def _m_pronoms_ton(self, sentence_str):
        return self._choose_random(sentence_str, ['ton', 'ta', 'tes'])

    def _m_mistake_onne(self, sentence_str):
        return self.replace_end_word(sentence_str, "onne", "", 2)

    def _m_mistake_on(self, sentence_str):
        return self.replace_end_word(sentence_str, "on", "ne", 0)

    def _m_parceque_mistake(self, sentence_str):
        return self.one_by_the_other(sentence_str, 'parce que', 'parce')

    def _m_ale_mistake(self, sentence_str):
        return self.one_by_the_other(sentence_str, ' au ', u'à le')

    def _m_mistake_ee(self, sentence_str):
        sentence = word_tokenize(sentence_str)
        rg = list(range(len(sentence)))
        random.shuffle(rg)
        found = False
        for i in rg:
            if sentence[i].endswith(u"é"):
                sentence[i] = sentence[i]+"e"
                found = True
            elif sentence[i].endswith(u"ée"):
                sentence[i] = sentence[i][:-1]
                found = False
        return found, self.detokenizer.detokenize(sentence, return_str=True)

    def _m_mistake_it(self, sentence_str):
        sentence = word_tokenize(sentence_str)
        rg = list(range(len(sentence)))
        random.shuffle(rg)
        found = False
        for i in rg:
            if sentence[i].endswith("i"):
                sentence[i] = sentence[i]+"it"
                found = True
            elif sentence[i].endswith("it"):
                sentence[i] = sentence[i][:-1]
                found = True
        return found, self.detokenizer.detokenize(sentence, return_str=True)

    def _m_mistake_it_bis(self, sentence_str):
        sentence = word_tokenize(sentence_str)
        rg = list(range(len(sentence)))
        random.shuffle(rg)
        found = False
        for i in rg:
            if sentence[i].endswith("i"):
                sentence[i] = sentence[i]+"is"
                found = True
            elif sentence[i].endswith("is"):
                sentence[i] = sentence[i][:-1]+"t"
                found = True
        return found, self.detokenizer.detokenize(sentence, return_str=True)

    def _m_mistake_es(self, sentence_str):
        sentence = word_tokenize(sentence_str)
        rg = list(range(len(sentence)))
        random.shuffle(rg)
        found = False
        for i in rg:
            if sentence[i].endswith(u"és"):
                sentence[i] = sentence[i][:-1]
                found = True
            elif sentence[i].endswith(u"é"):
                sentence[i] = sentence[i]+"s"
                found = True
        return found, self.detokenizer.detokenize(sentence, return_str=True)

    def _m_mistake_ees(self, sentence_str):
        sentence = word_tokenize(sentence_str)
        rg = list(range(len(sentence)))
        random.shuffle(rg)
        found = False
        for i in rg:
            if sentence[i].endswith(u"ées"):
                sentence[i] = sentence[i][:-1]
                found = True
            elif sentence[i].endswith(u"ée"):
                sentence[i] = sentence[i]+"s"
                found = True
        return found, self.detokenizer.detokenize(sentence, return_str=True)

    def _m_mistake_ez(self, sentence_str):
        sentence = word_tokenize(sentence_str)
        rg = list(range(len(sentence)))
        random.shuffle(rg)
        found = False
        for i in rg:
            if sentence[i].endswith("er"):
                sentence[i] = sentence[i][:-2]+"ez"
                found = True
            elif sentence[i].endswith("ez"):
                sentence[i] = sentence[i][:-2]+u"é"
                found = True
        return found, self.detokenizer.detokenize(sentence, return_str=True)

    def _m_mistake_eu(self, sentence_str):
        sentence = word_tokenize(sentence_str)
        rg = list(range(len(sentence)))
        random.shuffle(rg)
        found = False
        for i in rg:
            if sentence[i].endswith("eu"):
                sentence[i] = sentence[i]+"s"
                found = True
        return found, self.detokenizer.detokenize(sentence, return_str=True)

    def _m_mistake_ont(self, sentence_str):
        sentence = word_tokenize(sentence_str)
        rg = list(range(len(sentence)))
        random.shuffle(rg)
        found = False
        for i in rg:
            if sentence[i].endswith("ont"):
                sentence[i] = sentence[i][:-1]+"s"
                found = True
        return found, self.detokenizer.detokenize(sentence, return_str=True)

    def _m_mistake_eux(self, sentence_str):
        sentence = word_tokenize(sentence_str)
        rg = list(range(len(sentence)))
        random.shuffle(rg)
        found = False
        for i in rg:
            if sentence[i].endswith("eux"):
                sentence[i] = sentence[i][:-1]
                found = True
        return found, self.detokenizer.detokenize(sentence, return_str=True)

    def _m_mistake_eux_bis(self, sentence_str):
        sentence = word_tokenize(sentence_str)
        rg = list(range(len(sentence)))
        random.shuffle(rg)
        found = False
        for i in rg:
            if sentence[i].endswith("eux"):
                sentence[i] = sentence[i][:-1]+"t"
                found = True
        return found, self.detokenizer.detokenize(sentence, return_str=True)

    def _m_mistake_ait(self, sentence_str):
        sentence = word_tokenize(sentence_str)
        rg = list(range(len(sentence)))
        random.shuffle(rg)
        found = False
        for i in rg:
            if sentence[i].endswith("ait"):
                sentence[i] = sentence[i][:-1]+"ent"
                found = True
            elif sentence[i].endswith("aient"):
                sentence[i] = sentence[i][:-3]+"t"
                found = True
        return found, self.detokenizer.detokenize(sentence, return_str=True)

    def _m_mistake_o(self, sentence_str):
        if sentence_str.find(u'ô') > 0:
            return True, sentence_str.replace(u'ô', 'o', 1)
        else:
            return False, sentence_str

    def _m_mistake_u(self, sentence_str):
        if sentence_str.find(u'û') > 0:
            return True, sentence_str.replace(u'û', u'u', 1)
        else:
            return False, sentence_str

    def _m_mistake_u2(self, sentence_str):
        if sentence_str.find(u'ù') > 0:
            return True, sentence_str.replace(u'ù', 'u', 1)
        else:
            return False, sentence_str

    def _m_mistake_ai(self, sentence_str):
        sentence = word_tokenize(sentence_str)
        rg = list(range(len(sentence)))
        random.shuffle(rg)
        found = False
        for i in rg:
            if sentence[i].endswith("ait"):
                sentence[i] = sentence[i][:-1]
                found = True
            elif sentence[i].endswith("ais"):
                sentence[i] = sentence[i][:-1]
                found = True
        return found, self.detokenizer.detokenize(sentence, return_str=True)

    def _m_mistake_ent(self, sentence_str):
        sentence = word_tokenize(sentence_str)
        rg = list(range(len(sentence)))
        random.shuffle(rg)
        found = False
        for i in rg:
            if sentence[i].endswith("ent"):
                sentence[i] = sentence[i][:-2]
                found = True
        return found, self.detokenizer.detokenize(sentence, return_str=True)

    def _m_pronoms_quel(self, sentence_str):
        return self._choose_random(sentence_str, ['quel', 'quels', 'quelle', 'quelles'])

    def _m_pronoms_quelque(self, sentence_str):
        return self._choose_random(sentence_str, ['quelque', 'quelques'])

    def _m_mistake_plupart(self, sentence_str):
        return self._choose_random(sentence_str, ['plupart', 'la plus part'])

    def _m_mistake_demi(self, sentence_str):
        return self._choose_random(sentence_str, ['demi', 'demie'])

    def _m_mistake_nul(self, sentence_str):
        return self._choose_random(sentence_str, ['nul', 'nulle'])

    def _m_mistake_tout(self, sentence_str):
        return self._choose_random(sentence_str, ['tout', 'toutes', 'toute', 'tous'])

    def _m_mistake_le_la_les(self, sentence_str):
        return self._choose_random(sentence_str, [' le ', ' la '])

    def _m_mistake_un_une(self, sentence_str):
        return self._choose_random(sentence_str, [' un ', ' une '])

    def _m_mistake_leur(self, sentence_str):
        return self._choose_random(sentence_str, ['leur', 'leurs'])

    def _m_mistake_son(self, sentence_str):
        return self._choose_random(sentence_str, ['son', 'sa', 'ses', 'ces'])

    def _m_mistake_ces(self, sentence_str):
        return self._choose_random(sentence_str, ["c'est", "s'est", "ces", "ses"])

    def _m_mistake_de_du(self, sentence_str):
        return self.one_by_the_other(sentence_str, ' de ', ' du ')

    def _m_mistake_mm(self, sentence_str):
        return self.one_by_the_other(sentence_str, 'm', 'mm')

    def _m_mistake_mm(self, sentence_str):
        return self.one_by_the_other(sentence_str, 'mm', 'm')

    def _m_mistake_pp(self, sentence_str):
        return self.one_by_the_other(sentence_str, 'p', 'pp')

    def _m_mistake_pp_bis(self, sentence_str):
        return self.one_by_the_other(sentence_str, 'pp', 'p')

    def _m_mistake_que(self, sentence_str):
        return self.one_by_the_other(sentence_str, 'que', 'ce')

    def _m_mistake_en(self, sentence_str):
        return self.one_by_the_other(sentence_str, 'en', 'an')

    def _m_mistake_th(self, sentence_str):
        return self.one_by_the_other(sentence_str, 'th', 't')

    def _m_mistake_gg(self, sentence_str):
        return self.one_by_the_other(sentence_str, 'gg', 'g')

    def _m_mistake_rr(self, sentence_str):
        return self.one_by_the_other(sentence_str, 'rr', 'r')

    def _m_mistake_ff(self, sentence_str):
        return self.one_by_the_other(sentence_str, 'ff', 'f')

    def _m_mistake_ph(self, sentence_str):
        return self.one_by_the_other(sentence_str, 'ph', 'f')

    def _m_mistake_ch(self, sentence_str):
        return self.one_by_the_other(sentence_str, 'ch', 'c')

    def _m_mistake_ss(self, sentence_str):
        return self.one_by_the_other(sentence_str, 'ss', 's')

    def _m_mistake_ff(self, sentence_str):
        return self.one_by_the_other(sentence_str, 'ff', 'f')

    def _m_mistake_ll(self, sentence_str):
        return self.one_by_the_other(sentence_str, 'll', 'l')

    def _m_starts_with_h(self, sentence_str):
        sentence = word_tokenize(sentence_str)
        rg = list(range(len(sentence)))
        random.shuffle(rg)
        found = False
        for i in rg:
            if sentence[i].startswith('h') > 0:
                sentence[i] = sentence[i][1:]
                found = True
        return found, self.detokenizer.detokenize(sentence, return_str=True)

    def _m_ends_with_x(self, sentence_str):
        sentence = word_tokenize(sentence_str)
        rg = list(range(len(sentence)))
        random.shuffle(rg)
        found = False
        for i in rg:
            if sentence[i].endswith('x') > 0:
                sentence[i] = sentence[i][:-1]
                found = True
        return found, self.detokenizer.detokenize(sentence, return_str=True)


if __name__ == '__main__':
    mistake = Mistake()
    #print(mistake.methods)
    #mistakes = mistake.generate_mistakes(u'L\'entreprise est initialement une société de commerce de gros et de détail, principalement de tissus mais également de produits de quincaillerie3. Le développement du jean transformera radicalement la compagnie.')
    #for m in mistakes:
    #    print(m.encode('utf-8'))
    mistakes = mistake.generate_mistake(u'L\'entreprise est initialement une société de commerce de gros et de détail, principalement de tissus mais également de produits de quincaillerie3. Le développement du jean transformera radicalement la compagnie.')
    print(mistakes.encode('utf-8'))
