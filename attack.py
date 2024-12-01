import textattack
from textattack import Attack
from textattack.attack_recipes import AttackRecipe

from textattack.constraints.grammaticality import PartOfSpeech
from textattack.constraints.pre_transformation import (InputColumnModification, RepeatModification,StopwordModification)
from textattack.constraints.semantics.sentence_encoders import UniversalSentenceEncoder
from textattack.constraints.semantics import WordEmbeddingDistance
# An untargeted attack on classification models which attempts to minimize
#the score of the correct label until it is no longer the predicted label.
from textattack.goal_functions import UntargetedClassification
from textattack.search_methods import GreedyWordSwapWIR, GreedySearch
from textattack.transformations import WordInsertionMaskedLM, WordSwapMaskedLM, WordSwapEmbedding

class attackrecipe(AttackRecipe):


    def textfooler(model_wrapper):
        # Transforms an input by replacing its words with synonyms in the word embedding space.
        transformation = WordSwapEmbedding(max_candidates=50)
        stopwords = set(
            ["a", "about", "above", "across", "after", "afterwards", "again", "against", "ain", "all", "almost", "alone", "along", "already", "also", "although", "am", "among", "amongst", "an", "and", "another", "any", "anyhow", "anyone", "anything", "anyway", "anywhere", "are", "aren", "aren't", "around", "as", "at", "back", "been", "before", "beforehand", "behind", "being", "below", "beside", "besides", "between", "beyond", "both", "but", "by", "can", "cannot", "could", "couldn", "couldn't", "d", "didn", "didn't", "doesn", "doesn't", "don", "don't", "down", "due", "during", "either", "else", "elsewhere", "empty", "enough", "even", "ever", "everyone", "everything", "everywhere", "except", "first", "for", "former", "formerly", "from", "hadn", "hadn't", "hasn", "hasn't", "haven", "haven't", "he", "hence", "her", "here", "hereafter", "hereby", "herein", "hereupon", "hers", "herself", "him", "himself", "his", "how", "however", "hundred", "i", "if", "in", "indeed", "into", "is", "isn", "isn't", "it", "it's", "its", "itself", "just", "latter", "latterly", "least", "ll", "may", "me", "meanwhile", "mightn", "mightn't", "mine", "more", "moreover", "most", "mostly", "must", "mustn", "mustn't", "my", "myself", "namely", "needn", "needn't", "neither", "never", "nevertheless", "next", "no", "nobody", "none", "noone", "nor", "not", "nothing", "now", "nowhere", "o", "of", "off", "on", "once", "one", "only", "onto", "or", "other", "others", "otherwise", "our", "ours", "ourselves", "out", "over", "per", "please", "s", "same", "shan", "shan't", "she", "she's", "should've", "shouldn", "shouldn't", "somehow", "something", "sometime", "somewhere", "such", "t", "than", "that", "that'll", "the", "their", "theirs", "them", "themselves", "then", "thence", "there", "thereafter", "thereby", "therefore", "therein", "thereupon", "these", "they", "this", "those", "through", "throughout", "thru", "thus", "to", "too", "toward", "towards", "under", "unless", "until", "up", "upon", "used", "ve", "was", "wasn", "wasn't", "we", "were", "weren", "weren't", "what", "whatever", "when", "whence", "whenever", "where", "whereafter", "whereas", "whereby", "wherein", "whereupon", "wherever", "whether", "which", "while", "whither", "who", "whoever", "whole", "whom", "whose", "why", "with", "within", "without", "won", "won't", "would", "wouldn", "wouldn't", "y", "yet", "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves"]
        )
        #RepeatModification: A constraint disallowing the modification of words which have already been modified.
        #StopwordModification: A constraint disallowing the modification of stopwords.
        constraints = [RepeatModification(), StopwordModification(stopwords=stopwords)]
        # A constraint disallowing the modification of words within a specific input column.
        input_column_modification = InputColumnModification(
            ["premise", "hypothesis"], {"premise"}
        )
        constraints.append(input_column_modification)
        # A constraint on word substitutions which places a maximum distance between 
        # the embedding of the word being deleted and the word being inserted.
        constraints.append(WordEmbeddingDistance(min_cos_sim=0.8))
        # Constraints word swaps to only swap words with the same part of speech.
        constraints.append(PartOfSpeech(allow_verb_noun_swap=False))
        # Constraint using similarity between sentence encodings of x and x_adv
        # where the text embeddings are created using the Universal Sentence
        # Encoder.
        use_constraint = UniversalSentenceEncoder(
            threshold=0.936338023,
            metric="cosine",
            compare_against_original=False,
            window_size=15,
            skip_text_shorter_than_window=True,
        )
        constraints.append(use_constraint)
        goal_function = UntargetedClassification(model_wrapper)
        """
            GreedyWordSwapWIR: An attack that greedily chooses from a list of possible perturbations in
            order of index, after ranking indices by importance.

            Args:
            wir_method: method for ranking most important words
            model_wrapper: model wrapper used for gradient-based ranking
        """
        search_method = GreedyWordSwapWIR(wir_method="delete")
        return Attack(goal_function, constraints, transformation, search_method)
    
    def bertR(model_wrapper):
        # Generate potential replacements for a word using a masked language model.

        transformation = WordSwapMaskedLM(
            method="bae", max_candidates=50 #, min_confidence=0.0
        )
        constraints = [RepeatModification(), StopwordModification()]
        constraints.append(PartOfSpeech(allow_verb_noun_swap=False))
        use_constraint = UniversalSentenceEncoder(
            threshold=0.936338023,
            metric="cosine",
            compare_against_original=True,
            window_size=15,
            skip_text_shorter_than_window=True,
        )
        constraints.append(use_constraint)
        goal_function = UntargetedClassification(model_wrapper)
        search_method = GreedyWordSwapWIR(wir_method="delete")
        return Attack(goal_function, constraints, transformation, search_method)
    
    def bertI(model_wrapper):
        transformation = WordInsertionMaskedLM()
        constraints = [RepeatModification(), StopwordModification()]
        #constraints.append(PartOfSpeech(allow_verb_noun_swap=True))
        use_constraint = UniversalSentenceEncoder(
            threshold=0.936338023,
            metric="cosine",
            compare_against_original=True,
            window_size=15,
            skip_text_shorter_than_window=True,
        )
        constraints.append(use_constraint)
        goal_function = UntargetedClassification(model_wrapper)
        search_method = GreedyWordSwapWIR(wir_method="delete")
        return Attack(goal_function, constraints, transformation, search_method)
    
    def bertIR(model_wrapper):
        transformations = [WordSwapMaskedLM(method="bae", max_candidates=50, min_confidence=0.0),WordInsertionMaskedLM()]
        transformation = textattack.transformations.composite_transformation.CompositeTransformation(transformations)
        constraints = [RepeatModification(), StopwordModification()]
        constraints.append(PartOfSpeech(allow_verb_noun_swap=False, compare_against_original=False))
        use_constraint = UniversalSentenceEncoder(
            threshold=0.936338023,
            metric="cosine",
            compare_against_original=True,
            window_size=15,
            skip_text_shorter_than_window=True,
        )
        constraints.append(use_constraint)
        goal_function = UntargetedClassification(model_wrapper)
        search_method = GreedyWordSwapWIR(wir_method="delete")
        return Attack(goal_function, constraints, transformation, search_method)


