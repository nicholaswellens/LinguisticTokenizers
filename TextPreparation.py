import datetime
import time

import collections

import pickle
import numpy as np
from frog import Frog, FrogOptions
import ucto
from polyglot.text import Text, Word
from numpy import asarray, savetxt, loadtxt
from string import punctuation
#import tkinter
#import _tkinter
import pandas as pd
#import matplotlib
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import morfessor


def format_time(elapsed):
  '''
  Takes a time in seconds and returns a string hh:mm:ss
  '''
  # Round to the nearest second.
  elapsed_rounded = int(round((elapsed)))

  # Format as hh:mm:ss
  return str(datetime.timedelta(seconds=elapsed_rounded))


## Taken from https://stackoverflow.com/questions/5920643/add-an-item-between-each-item-already-in-the-list
def intersperse(lst, item):
    result = [item] * (len(lst) * 2 - 1)
    result[0::2] = lst
    return result

print("TESTESSTTTT")
print(intersperse(["mee","speel","en","de"],"__ADD_MERGE__"))
print(intersperse(["ik"],"__ADD_MERGE__"))

configurationFile = "tokconfig-nld"
tokenizer = ucto.Tokenizer(configurationFile)

def ucto_tokenize(sentence):

    tokenized_sentence = []
    tokenizer.process(sentence)
    for token in tokenizer:
      tokenized_sentence += [str(token)]
    return tokenized_sentence

def change_text_to_morphs(sentences, frog_merge = False,  save = False, filename=None):
    # sentence list to sentence list in frog morphism form
    morphSentences = []

    frog = Frog(
        FrogOptions(tok=True, lemma=True, morph=True, daringmorph=False, mwu=False, chunking=False, ner=False,
                    parser=False))

    for sentenceNumber in range(0,len(sentences)):
        print(sentenceNumber)
        print("of")
        print(len(sentences))
        sentenceToBeProcessed = sentences[sentenceNumber]
        sentenceToBeProcessed = sentenceToBeProcessed.replace("\n"," ")
        morphSentence = []
        output = frog.process(sentenceToBeProcessed)
        for i in range(0,len(output)):
            morphisms_word = output[i].get("morph")
            #print(morphisms_word[0:-1])
            morphisms_word_list = morphisms_word.replace('[', '').split(']')
            morphisms_word_list = list(filter(None, morphisms_word_list)) #For lettergreep there are ' ' it seems, so need to remove
            #print(morphisms_word_list)
            if frog_merge:
                morphisms_word = list(filter(None, morphisms_word_list))
                morphisms_word_list = intersperse(morphisms_word_list, "[MERGE]")

            print(morphisms_word_list[0])
            if morphisms_word_list[0] == "93434343402020219293949001019298":
                print("HEPPEERE")
                morphisms_word_list = ["[MASK]"]
            #print("EVET")
            #print(morphisms_word_list)
            morphSentence += morphisms_word_list
        #print("MORPHSENTENCE")
        #print(morphSentence)
        # Remove the empty strings
        morphSentence = list(filter(None, morphSentence))
        morphSentence = ' '.join(morphSentence)
        #print("HERE")
        #print(morphSentence)
        morphSentences.append(morphSentence)

    if save is True:
        with open(filename, 'wb') as outputfile:
            pickle.dump(morphSentences, outputfile)
    return morphSentences


def convertToPolyglotMorf(sentences, save = False, merge=False,filename = "noname.pickle" ):
    #List of str to List of str (morfemes)

    total_review_morf_text_list = []
    i = 1
    morfed_sentences = []
    print(len(sentences))
    for sentence in sentences:
      sentence = sentence.replace("\n", " ")
      print(i)
      print("of")
      print(len(sentences))
      tokenized_sentence = ucto_tokenize(sentence)
      morfed_sentence = []
      for w in tokenized_sentence:
        w = str(w)
        w = Word(w, language="nl")
        word_morphemes = w.morphemes
        word_morphemes = list(filter(None, word_morphemes))
        #print("{:<20}{}".format(w, w.morphemes))
        if merge is True:
            morfed_sentence += intersperse(w.morphemes, "[MERGE]")
        else:
            morfed_sentence += w.morphemes

      morfed_sentence = ' '.join(morfed_sentence)

      #print(review_morf_list)
      morfed_sentences.append(morfed_sentence)
      i+=1

    if save is True:
        with open(filename, 'wb') as outputfile:
            pickle.dump(morfed_sentences, outputfile)

    return morfed_sentences


def convertToMorfessorMorf(sentences, save = False, merge=False,filename = "noname.pickle"):
    len_sentences = len(sentences)

    io = morfessor.MorfessorIO()

    model = io.read_binary_model_file('TrainFiles/Officialmodel.bin')
    morfed_sentences = []
    i = 1
    for sentenceToBeProcessed in sentences:
        print(i)
        print("of")
        print(len_sentences)
        i += 1
        # print(sentenceToBeProcessed)
        # print("--------------------------------------------------")
        sentenceToBeProcessed = sentenceToBeProcessed.replace("\n", " ")
        tokenized_sentence = ucto_tokenize(sentenceToBeProcessed)
        morfed_sentence = []
        for w in tokenized_sentence:
            word_morphemes = model.viterbi_segment(w)
            word_morphemes = list(filter(None, word_morphemes))
            # print(word_morphemes)
            # print(word_morphemes[0])
            if merge is True:
                morfed_sentence += intersperse(word_morphemes[0], "[MERGE]")
            else:
                morfed_sentence += word_morphemes[0]
        morfed_sentence = ' '.join(morfed_sentence)
        # print(morfed_sentence)
        # print("--------------------------------------------------")
        # print("--------------------------------------------------")
        # print("--------------------------------------------------")
        morfed_sentences.append(morfed_sentence)

        if save is True:
            with open(filename, 'wb') as outputfile:
                pickle.dump(morfed_sentences, outputfile)

    return morfed_sentences

def change_text_to_lemma_POS(sentences,  save = False, filename=None):
    # sentence list to sentence list in frog lemma + pos
    lemmapos_sentences = []

    frog = Frog(FrogOptions(tok=True, lemma=True, morph=False, daringmorph=False, mwu=False, chunking=False, ner=False,
                            parser=False))


    for sentenceNumber in range(0, len(sentences)):
        print(sentenceNumber)
        print("of")
        print(len(sentences))
        sentenceToBeProcessed = sentences[sentenceNumber]
        sentenceToBeProcessed = sentenceToBeProcessed.replace("\n", " ")

        output = frog.process(sentenceToBeProcessed)
        lemmapos_sentence = ""
        for i in range(0,len(output)):
            pos = str(output[i].get("pos"))
            lemma = str(output[i].get("lemma"))
            #posprob = str(output[i].get("posprob"))
            #print(posprob)

            # print("pos:      " + pos)
            # print("lemma:    " + lemma)

            pos = "<" + pos
            pos = pos.replace("(", "><")
            pos = pos.replace(")", ">")
            pos = pos.replace(",", "><")
            pos = pos.replace("<>", "")

            # print(pos)

            lemmapos_word = lemma + " " + pos

            if lemma == "93434343402020219293949001019298":
                lemmapos_word = "[MASK]"

            #word = str(output[i].get("text"))
            #print(f"{word}: {lemmapos_word}")

            lemmapos_sentence = lemmapos_sentence + " " + lemmapos_word

        # Remove the first empty string
        #print(lemmapos_sentence)

        lemmapos_sentence = lemmapos_sentence[1:]
        #print("")
        #print("")
        #print("")
        #print("")
        #print(lemmapos_sentence)
        #print("")
        #print("")
        #print("")
        #print("")
        lemmapos_sentences.append(lemmapos_sentence)
        #print("")
        #print(lemmapos_sentences)
        #print("")

    if save is True:
        with open(filename, 'wb') as outputfile:
            pickle.dump(lemmapos_sentences, outputfile)
    return lemmapos_sentences

#------------------------------Script for POLYGLOT MORFMERGE Sentiment-----------------------------

# with open('TrainFiles/totTrainListShuffled.data', 'rb') as filehandle:
#     # read the data as binary data stream
#     totTrainList = pickle.load(filehandle)
#
# with open('TrainFiles/totTestListShuffled.data', 'rb') as filehandle:
#     # read the data as binary data stream
#     totTestList = pickle.load(filehandle)

# with open('TrainFiles/totTrainListShuffled.pickle', 'wb') as outputfile:
#     pickle.dump(totTrainList, outputfile)
#
# with open('TrainFiles/totTestListShuffled.pickle', 'wb') as outputfile:
#     pickle.dump(totTestList, outputfile)

# #afterPunctuationMergeMorfedTrainSentiment = convertToPolyglotMorf(totTrainList[10000:], merge=True,save = True,filename="TrainFiles/afterPunctuationMergeMorfedTrainSentiment_Part2.pickle")
#
# ALL THIS UNDER IS BECAUSE IT WERE LIST OF LISTS AND SHOULD BE LIST OF STRINGS, BUT NOW covertToPolyglotMorf is UPDATED AND BETTER
#
# with open('TrainFiles/afterPunctuationMergeMorfedTrainSentiment.pickle', 'rb') as inputfile:
#     afterPunctuationMergeMorfedTrainSentiment_Part1 = pickle.load(inputfile)
# with open('TrainFiles/afterPunctuationMergeMorfedTrainSentiment_Part2.pickle', 'rb') as inputfile:
#     afterPunctuationMergeMorfedTrainSentiment_Part2 = pickle.load(inputfile)
#
# afterPunctuationMergeMorfedTrainSentiment = afterPunctuationMergeMorfedTrainSentiment_Part1 + afterPunctuationMergeMorfedTrainSentiment_Part2
#
# with open('TrainFiles/afterPunctuationMergeMorfedTestSentiment.pickle', 'rb') as inputfile:
#     afterPunctuationMergeMorfedTestSentiment = pickle.load(inputfile)
#
# print(afterPunctuationMergeMorfedTrainSentiment[:2])
#
# for i in range(len(afterPunctuationMergeMorfedTrainSentiment)):
#     afterPunctuationMergeMorfedTrainSentiment[i] = ' '.join(afterPunctuationMergeMorfedTrainSentiment[i])
# for i in range(len(afterPunctuationMergeMorfedTestSentiment)):
#     afterPunctuationMergeMorfedTestSentiment[i] = ' '.join(afterPunctuationMergeMorfedTestSentiment[i])
#
# print(afterPunctuationMergeMorfedTrainSentiment[:2])
#
# with open("TrainFiles/PolyglotMorfTrainReviewsMergeEdited.pickle", 'wb') as outputfile:
#     pickle.dump(afterPunctuationMergeMorfedTrainSentiment, outputfile)
#
# with open("TrainFiles/PolyglotMorfTestReviewsMergeEdited.pickle", 'wb') as outputfile:
#     pickle.dump(afterPunctuationMergeMorfedTestSentiment, outputfile)

#afterPunctuationMergeMorfedTestSentiment = convertToPolyglotMorf(totTestList, merge=True,save = True,filename="TrainFiles/afterPunctuationMergeMorfedTestSentiment.pickle")
# with open('TrainFiles/afterPunctuationMergeFrogMorphedLettergreep.pickle', 'rb') as inputfile:
#     afterPunctuationMergeFrogMorphedLettergreep = pickle.load(inputfile)





### Save new file

# morfed_totTrainList = convertToPolyglotMorf(totTrainList_Cleaned, save=True)
# Open = open('TrainFiles/convertedPolyglotMorfText.txt','r')
# morfed_totTrainList = Open.read()
# morfed_totTrainList = morfed_totTrainList.split('*%')
#

#------------------------------Script for FROG MORPHMERGE Sentiment NOT OFFICIAL MAYBE CHECK UNDER DIFFERENT ONE-----------------------------

# split_frac = 0.9 #splitting fraction between TRAIN and TEST data
#
# #------------------------------------------------------------
# #------------------------------------------------------------
# #------------------------------------------------------------
#
#
#
# with open('TrainFiles/FrogMorphedTrainLettergreep.pickle', 'rb') as inputfile:
#     FrogMorphedTrainLettergreep = pickle.load(inputfile)
#
# #FrogMorphedTestLettergreep = change_text_to_morphs(lettergreep_test_words,  save = True, filename = 'TrainFiles/FrogMorphedTestLettergreep.pickle')
# with open('TrainFiles/FrogMorphedTestLettergreep.pickle', 'rb') as inputfile:
#     FrogMorphedTestLettergreep = pickle.load(inputfile)
#
# print(FrogMorphedTrainLettergreep[:15])
#
# allFrogMorphedLettergreep = FrogMorphedTrainLettergreep + FrogMorphedTestLettergreep
#
# print("lennnn")
# print(len(allFrogMorphedLettergreep))
#
# data = pd.read_csv("TrainFiles/lettergrepen_Official3.csv")
# lettergreep_words = list(data["woorden"])
# lettergreep_splits = list(data["lettergrepen"])
# lettergreep_labels = list(data["aantal lettergrepen"])
#
# len_dataset = len(lettergreep_words)
# #
# #
# #
# #
# #
# #
# #
# # #-----------------Delete Examples with punctuation (both after FROG and in normal for EACH) ---------
# #
# ### Ree "#Name?" words..
# i=0
# while i <  len(lettergreep_words):
#     if i < len(lettergreep_words):
#         if lettergreep_words[i] == "#NAME?":
#             print("ok")
#             # NOTTT allFrogMorphedLettergreep.pop(i) BECAUSE ALREADY REMOVED
#             lettergreep_words.pop(i)
#             lettergreep_labels.pop(i)
#             lettergreep_splits.pop(i)
#         else:
#             i+=1
#
# print("HERE1: ")
# print(len(lettergreep_words))
# i=0
# while i < len(allFrogMorphedLettergreep):
#     if i < len(allFrogMorphedLettergreep):
#         containsFrog = [c for c in allFrogMorphedLettergreep[i] if c in punctuation]
#         containsNormal = [c for c in lettergreep_words[i] if c in punctuation]
#         if len(containsFrog) > 0 or len(containsNormal) > 0:
#             #print(contains)
#             allFrogMorphedLettergreep.pop(i)
#             lettergreep_words.pop(i)
#             lettergreep_labels.pop(i)
#             lettergreep_splits.pop(i)
#         else:
#             i+=1
#
# print("HERE2: ")
# print(len(lettergreep_words))
#
# #
# # for i in range(len(allFrogMorphedLettergreep)):
# #     if i < len(allFrogMorphedLettergreep):
# #         containsFrog = [c for c in allFrogMorphedLettergreep[i] if c in punctuation]
# #         containsNormal = [c for c in lettergreep_words[i] if c in punctuation]
# #         if lettergreep_words[i] == "jonko's":
# #             print("")
# #             print(lettergreep_words[i] )
# #             print("FOUND 1")
# #             print("")
# #         if len(containsFrog) > 0 or len(containsNormal) > 0:
# #             #print(contains)
# #             allFrogMorphedLettergreep.pop(i)
# #             lettergreep_words.pop(i)
# #             lettergreep_labels.pop(i)
# #             lettergreep_splits.pop(i)
#
# print(len(lettergreep_words))
# print("equal?")
# print(len(allFrogMorphedLettergreep))
#
# # for i in range(len(lettergreep_words)):
# #     if i < len(lettergreep_words):
# #         contains = [c for c in lettergreep_words[i] if c in punctuation]
# #
# #         if lettergreep_words[i] == "jonko's":
# #             print("")
# #             print(lettergreep_words[i] )
# #             print("FOUND 2")
# #             print("")
# #         if len(contains) > 0:
# #             #print(lettergreep_words[i])
# #             allFrogMorphedLettergreep.pop(i)
# #             lettergreep_words.pop(i)
# #             lettergreep_labels.pop(i)
# #             lettergreep_splits.pop(i)
#
# print("How manyyyyy with punctuation: ")
# j=0
# for i in range(len(lettergreep_words)):
#     if i < len(lettergreep_words):
#         contains = [c for c in lettergreep_words[i] if c in punctuation]
#         if len(contains) >0:
#             #print(lettergreep_words[i])
#             j+=1
# print(j)
#
# print("")
# print("How many with punctuation: ")
# i=0
# for word in lettergreep_words:
#     contains = [c for c in word if c in punctuation]
#     if len(contains) >0:
#         #print(word)
#         i+=1
# print(i)
# print("with")
#
#
# print(len(lettergreep_words))
# print("equal?")
# print(len(allFrogMorphedLettergreep))
#
#
# #------------------------------------------------------------------
#
#
# len_dataset = len(lettergreep_words)
# print(len_dataset)
# print(len(lettergreep_splits))
# print(len(lettergreep_labels))
#
#
# FrogMorphedTrainLettergreep = allFrogMorphedLettergreep[0:int(split_frac*len_dataset)]
# lettergreep_train_words = lettergreep_words[0:int(split_frac*len_dataset)]
# lettergreep_train_labels = lettergreep_labels[0:int(split_frac*len_dataset)]
#
# FrogMorphedTestLettergreep = allFrogMorphedLettergreep[int(split_frac*len_dataset):]
# lettergreep_test_words = lettergreep_words[int(split_frac*len_dataset):]
# lettergreep_test_labels = lettergreep_labels[int(split_frac*len_dataset):]
#
# #IPython ; IPython.embed() ; exit(1)
# print(len(allFrogMorphedLettergreep))
# print("Number of train examples")
# print(len(FrogMorphedTrainLettergreep))
# print("")
# print("Number of test examples")
# print(len(FrogMorphedTestLettergreep))
# #########################################
#
# print("Duplicates: ")
# print([item for item, count in collections.Counter(lettergreep_words).items()if count >1])
#
#
#
#
#
# print("How many with punctuation: ")
# i=0
# for word in lettergreep_words:
#     contains = [c for c in word if c in punctuation]
#     if len(contains) >0:
#         #print(word)
#         i+=1
#
# print(i)
# print("with")
# print(lettergreep_words[:10])
#
# afterPunctuationMergePolyglotMorfedLettergreep = convertToPolyglotMorf(lettergreep_words, merge=True,save = True,filename="TrainFiles/afterPunctuationMergePolyglotMorfedLettergreep.pickle")
# ##with open('TrainFiles/afterPunctuationMergePolyglotMorfedLettergreep.pickle', 'rb') as inputfile:
# ##    afterPunctuationMergePolyglotMorfedLettergreep = pickle.load(inputfile)
#
# print(afterPunctuationMergePolyglotMorfedLettergreep[:10])
# for i in range(len(afterPunctuationMergePolyglotMorfedLettergreep)):
#     afterPunctuationMergePolyglotMorfedLettergreep[i] = afterPunctuationMergePolyglotMorfedLettergreep[i].replace(" [MERGE] ", " ").replace(" [MERGE]", " ")
# print(afterPunctuationMergePolyglotMorfedLettergreep[:10])
# print("How many with punctuation MorphMerge: ")
#
# i=0
# for word in afterPunctuationMergePolyglotMorfedLettergreep:
#     contains = [c for c in word if c in punctuation]
#     if len(contains) >0:
#         #print(word)
#         i+=1
# print(i)
# print("with")
# print(f"length: {len(afterPunctuationMergePolyglotMorfedLettergreep)}")
#
# print("")
# print("ANALYSIS ANALYSIS")
# print(f"length of lettergreep_words: {len(lettergreep_words)}")
# print(f"length of afterPunctuationMergeFrogMorphedLettergreep: {len(afterPunctuationMergePolyglotMorfedLettergreep)}")
# print("")
# print(afterPunctuationMergePolyglotMorfedLettergreep[234])
# print(lettergreep_words[234])
# print(afterPunctuationMergePolyglotMorfedLettergreep[1234])
# print(lettergreep_words[1234])
# print(afterPunctuationMergePolyglotMorfedLettergreep[11234])
# print(lettergreep_words[11234])
# print(afterPunctuationMergePolyglotMorfedLettergreep[500])
# print(lettergreep_words[500])
# print(afterPunctuationMergePolyglotMorfedLettergreep[546:556])
# print(lettergreep_words[546:556])
# print("FINISHED FINISHED ANALYSIS ANALYSIS")




#------------------------------Script for MORPH lettergreep-----------------------------

# split_frac = 0.9 #splitting fraction between TRAIN and TEST data
#
# #------------------------------------------------------------
# #------------------------------------------------------------
# #------------------------------------------------------------
#
#
#
# with open('TrainFiles/FrogMorphedTrainLettergreep.pickle', 'rb') as inputfile:
#     FrogMorphedTrainLettergreep = pickle.load(inputfile)
#
# #FrogMorphedTestLettergreep = change_text_to_morphs(lettergreep_test_words,  save = True, filename = 'TrainFiles/FrogMorphedTestLettergreep.pickle')
# with open('TrainFiles/FrogMorphedTestLettergreep.pickle', 'rb') as inputfile:
#     FrogMorphedTestLettergreep = pickle.load(inputfile)
#
# print(FrogMorphedTrainLettergreep[:15])
#
# allFrogMorphedLettergreep = FrogMorphedTrainLettergreep + FrogMorphedTestLettergreep
#
# print("lennnn")
# print(len(allFrogMorphedLettergreep))
#
# data = pd.read_csv("TrainFiles/lettergrepen_Official3.csv")
# lettergreep_words = list(data["woorden"])
# lettergreep_splits = list(data["lettergrepen"])
# lettergreep_labels = list(data["aantal lettergrepen"])
#
# len_dataset = len(lettergreep_words)
# #
# #
# #
# #
# #
# #
# #
# # #-----------------Delete Examples with punctuation (both after FROG and in normal for EACH) ---------
# #
# ### Ree "#Name?" words..
# i=0
# while i <  len(lettergreep_words):
#     if i < len(lettergreep_words):
#         if lettergreep_words[i] == "#NAME?":
#             print("ok")
#             # NOTTT allFrogMorphedLettergreep.pop(i) BECAUSE ALREADY REMOVED
#             lettergreep_words.pop(i)
#             lettergreep_labels.pop(i)
#             lettergreep_splits.pop(i)
#         else:
#             i+=1
#
# print("HERE1: ")
# print(len(lettergreep_words))
# i=0
# while i < len(allFrogMorphedLettergreep):
#     if i < len(allFrogMorphedLettergreep):
#         containsFrog = [c for c in allFrogMorphedLettergreep[i] if c in punctuation]
#         containsNormal = [c for c in lettergreep_words[i] if c in punctuation]
#         if len(containsFrog) > 0 or len(containsNormal) > 0:
#             #print(contains)
#             allFrogMorphedLettergreep.pop(i)
#             lettergreep_words.pop(i)
#             lettergreep_labels.pop(i)
#             lettergreep_splits.pop(i)
#         else:
#             i+=1
#
# print("HERE2: ")
# print(len(lettergreep_words))
#
# #
# # for i in range(len(allFrogMorphedLettergreep)):
# #     if i < len(allFrogMorphedLettergreep):
# #         containsFrog = [c for c in allFrogMorphedLettergreep[i] if c in punctuation]
# #         containsNormal = [c for c in lettergreep_words[i] if c in punctuation]
# #         if lettergreep_words[i] == "jonko's":
# #             print("")
# #             print(lettergreep_words[i] )
# #             print("FOUND 1")
# #             print("")
# #         if len(containsFrog) > 0 or len(containsNormal) > 0:
# #             #print(contains)
# #             allFrogMorphedLettergreep.pop(i)
# #             lettergreep_words.pop(i)
# #             lettergreep_labels.pop(i)
# #             lettergreep_splits.pop(i)
#
# print(len(lettergreep_words))
# print("equal?")
# print(len(allFrogMorphedLettergreep))
#
# # for i in range(len(lettergreep_words)):
# #     if i < len(lettergreep_words):
# #         contains = [c for c in lettergreep_words[i] if c in punctuation]
# #
# #         if lettergreep_words[i] == "jonko's":
# #             print("")
# #             print(lettergreep_words[i] )
# #             print("FOUND 2")
# #             print("")
# #         if len(contains) > 0:
# #             #print(lettergreep_words[i])
# #             allFrogMorphedLettergreep.pop(i)
# #             lettergreep_words.pop(i)
# #             lettergreep_labels.pop(i)
# #             lettergreep_splits.pop(i)
#
# print("How manyyyyy with punctuation: ")
# j=0
# for i in range(len(lettergreep_words)):
#     if i < len(lettergreep_words):
#         contains = [c for c in lettergreep_words[i] if c in punctuation]
#         if len(contains) >0:
#             #print(lettergreep_words[i])
#             j+=1
# print(j)
#
# print("")
# print("How many with punctuation: ")
# i=0
# for word in lettergreep_words:
#     contains = [c for c in word if c in punctuation]
#     if len(contains) >0:
#         #print(word)
#         i+=1
# print(i)
# print("with")
#
#
# print(len(lettergreep_words))
# print("equal?")
# print(len(allFrogMorphedLettergreep))
#
#
# #------------------------------------------------------------------
#
#
# len_dataset = len(lettergreep_words)
# print(len_dataset)
# print(len(lettergreep_splits))
# print(len(lettergreep_labels))
#
#
# FrogMorphedTrainLettergreep = allFrogMorphedLettergreep[0:int(split_frac*len_dataset)]
# lettergreep_train_words = lettergreep_words[0:int(split_frac*len_dataset)]
# lettergreep_train_labels = lettergreep_labels[0:int(split_frac*len_dataset)]
#
# FrogMorphedTestLettergreep = allFrogMorphedLettergreep[int(split_frac*len_dataset):]
# lettergreep_test_words = lettergreep_words[int(split_frac*len_dataset):]
# lettergreep_test_labels = lettergreep_labels[int(split_frac*len_dataset):]
#
# #IPython ; IPython.embed() ; exit(1)
# print(len(allFrogMorphedLettergreep))
# print("Number of train examples")
# print(len(FrogMorphedTrainLettergreep))
# print("")
# print("Number of test examples")
# print(len(FrogMorphedTestLettergreep))
# #########################################
#
# print("Duplicates: ")
# print([item for item, count in collections.Counter(lettergreep_words).items()if count >1])
#
#
#
#
#
# print("How many with punctuation: ")
# i=0
# for word in lettergreep_words:
#     contains = [c for c in word if c in punctuation]
#     if len(contains) >0:
#         #print(word)
#         i+=1
#
# print(i)
# print("with")
# print(lettergreep_words[:10])
#
# #afterPunctuationMergeFrogMorphedLettergreep = change_text_to_morphs(lettergreep_words, frog_merge=True,save = True,filename="TrainFiles/afterPunctuationMergeFrogMorphedLettergreep.pickle")
# with open('TrainFiles/afterPunctuationMergeFrogMorphedLettergreep.pickle', 'rb') as inputfile:
#     afterPunctuationMergeFrogMorphedLettergreep = pickle.load(inputfile)
#
# print(afterPunctuationMergeFrogMorphedLettergreep[:10])
# for i in range(len(afterPunctuationMergeFrogMorphedLettergreep)):
#     afterPunctuationMergeFrogMorphedLettergreep[i] = afterPunctuationMergeFrogMorphedLettergreep[i].replace(" [MERGE] ", " ").replace(" [MERGE]", " ")
# print(afterPunctuationMergeFrogMorphedLettergreep[:10])
# print("How many with punctuation MorphMerge: ")
#
# i=0
# for word in afterPunctuationMergeFrogMorphedLettergreep:
#     contains = [c for c in word if c in punctuation]
#     if len(contains) >0:
#         #print(word)
#         i+=1
# print(i)
# print("with")
# print(f"length: {len(afterPunctuationMergeFrogMorphedLettergreep)}")
#
# print("")
# print("ANALYSIS ANALYSIS")
# print(f"length of lettergreep_words: {len(lettergreep_words)}")
# print(f"length of afterPunctuationMergeFrogMorphedLettergreep: {len(afterPunctuationMergeFrogMorphedLettergreep)}")
# print("")
# print(afterPunctuationMergeFrogMorphedLettergreep[234])
# print(lettergreep_words[234])
# print(afterPunctuationMergeFrogMorphedLettergreep[1234])
# print(lettergreep_words[1234])
# print(afterPunctuationMergeFrogMorphedLettergreep[11234])
# print(lettergreep_words[11234])
# print(afterPunctuationMergeFrogMorphedLettergreep[500])
# print(lettergreep_words[500])
# print(afterPunctuationMergeFrogMorphedLettergreep[546:556])
# print(lettergreep_words[546:556])
# print("FINISHED FINISHED ANALYSIS ANALYSIS")
#
#
#
#
#

###########################################################################################
#------------------------------Script for POLYGLOT MORFESSOR LEMMAPOS (but adjust a bit) lettergreep-----------------------------
###########################################################################################

# split_frac = 0.9 #splitting fraction between TRAIN and TEST data
#
# #------------------------------------------------------------
# #------------------------------------------------------------
# #------------------------------------------------------------
#
#
#
# with open('TrainFiles/FrogMorphedTrainLettergreep.pickle', 'rb') as inputfile:
#     FrogMorphedTrainLettergreep = pickle.load(inputfile)
#
# #FrogMorphedTestLettergreep = change_text_to_morphs(lettergreep_test_words,  save = True, filename = 'TrainFiles/FrogMorphedTestLettergreep.pickle')
# with open('TrainFiles/FrogMorphedTestLettergreep.pickle', 'rb') as inputfile:
#     FrogMorphedTestLettergreep = pickle.load(inputfile)
#
# print(FrogMorphedTrainLettergreep[:15])
#
# allFrogMorphedLettergreep = FrogMorphedTrainLettergreep + FrogMorphedTestLettergreep
#
# print("lennnn")
# print(len(allFrogMorphedLettergreep))
#
# data = pd.read_csv("TrainFiles/lettergrepen_Official3.csv")
# lettergreep_words = list(data["woorden"])
# lettergreep_splits = list(data["lettergrepen"])
# lettergreep_labels = list(data["aantal lettergrepen"])
#
# len_dataset = len(lettergreep_words)
# #
# #
# #
# #
# #
# #
# #
# # #-----------------Delete Examples with punctuation (both after FROG and in normal for EACH) ---------
# #
# ### Ree "#Name?" words..
# i=0
# while i <  len(lettergreep_words):
#     if i < len(lettergreep_words):
#         if lettergreep_words[i] == "#NAME?":
#             print("ok")
#             # NOTTT allFrogMorphedLettergreep.pop(i) BECAUSE ALREADY REMOVED
#             lettergreep_words.pop(i)
#             lettergreep_labels.pop(i)
#             lettergreep_splits.pop(i)
#         else:
#             i+=1
#
# print("HERE1: ")
# print(len(lettergreep_words))
# i=0
# while i < len(allFrogMorphedLettergreep):
#     if i < len(allFrogMorphedLettergreep):
#         containsFrog = [c for c in allFrogMorphedLettergreep[i] if c in punctuation]
#         containsNormal = [c for c in lettergreep_words[i] if c in punctuation]
#         if len(containsFrog) > 0 or len(containsNormal) > 0:
#             #print(contains)
#             allFrogMorphedLettergreep.pop(i)
#             lettergreep_words.pop(i)
#             lettergreep_labels.pop(i)
#             lettergreep_splits.pop(i)
#         else:
#             i+=1
#
# print("HERE2: ")
# print(len(lettergreep_words))
#
# print(len(lettergreep_words))
# print("equal?")
# print(len(allFrogMorphedLettergreep))
#
#
# print("How manyyyyy with punctuation: ")
# j=0
# for i in range(len(lettergreep_words)):
#     if i < len(lettergreep_words):
#         contains = [c for c in lettergreep_words[i] if c in punctuation]
#         if len(contains) >0:
#             #print(lettergreep_words[i])
#             j+=1
# print(j)
#
# print("")
# print("How many with punctuation: ")
# i=0
# for word in lettergreep_words:
#     contains = [c for c in word if c in punctuation]
#     if len(contains) >0:
#         #print(word)
#         i+=1
# print(i)
# print("with")
#
#
#
# print(len(lettergreep_words))
# print("equal?")
# print(len(allFrogMorphedLettergreep))
#
#
# #------------------------------------------------------------------
#
#
# len_dataset = len(lettergreep_words)
# print(len_dataset)
# print(len(lettergreep_splits))
# print(len(lettergreep_labels))
#
#
# FrogMorphedTrainLettergreep = allFrogMorphedLettergreep[0:int(split_frac*len_dataset)]
# lettergreep_train_words = lettergreep_words[0:int(split_frac*len_dataset)]
# lettergreep_train_labels = lettergreep_labels[0:int(split_frac*len_dataset)]
#
# FrogMorphedTestLettergreep = allFrogMorphedLettergreep[int(split_frac*len_dataset):]
# lettergreep_test_words = lettergreep_words[int(split_frac*len_dataset):]
# lettergreep_test_labels = lettergreep_labels[int(split_frac*len_dataset):]
#
# #IPython ; IPython.embed() ; exit(1)
# print(len(allFrogMorphedLettergreep))
# print("Number of train examples")
# print(len(FrogMorphedTrainLettergreep))
# print("")
# print("Number of test examples")
# print(len(FrogMorphedTestLettergreep))
# #########################################
#
# print("Duplicates: ")
# print([item for item, count in collections.Counter(lettergreep_words).items()if count >1])
#
#
#
#
#
# print("How many with punctuation: ")
# i=0
# for word in lettergreep_words:
#     contains = [c for c in word if c in punctuation]
#     if len(contains) >0:
#         #print(word)
#         i+=1
#
# print(i)
# print("with")
# print(lettergreep_words[:10])
#
# #afterPunctuationLemmaPosLettergreep = change_text_to_lemma_POS(lettergreep_words, save = True,filename="TrainFiles/afterPunctuationLemmaPosLettergreep.pickle")#change_text_to_morphs(lettergreep_words, frog_merge=True,save = True,filename="TrainFiles/afterPunctuationMergeFrogMorphedLettergreep.pickle")
# with open('TrainFiles/afterPunctuationLemmaPosLettergreep.pickle', 'rb') as inputfile:
#     afterPunctuationLemmaPosLettergreep = pickle.load(inputfile)
#
# print(afterPunctuationLemmaPosLettergreep)
# # for i in range(len(afterPunctuationLemmaPosLettergreep)):
# #     afterPunctuationLemmaPosLettergreep[i] = afterPunctuationLemmaPosLettergreep[i].replace(" [MERGE] ", " ").replace(" [MERGE]", " ")
# # print(afterPunctuationLemmaPosLettergreep[:10])
# # print("How many with punctuation MorphMerge: ")
# #
# # i=0
# # for word in afterPunctuationLemmaPosLettergreep:
# #     contains = [c for c in word if c in punctuation]
# #     if len(contains) >0:
# #         #print(word)
# #         i+=1
# # print(i)
# print("with")
# print(f"length: {len(afterPunctuationLemmaPosLettergreep)}")
#
# print("")
# print("ANALYSIS ANALYSIS")
# print(f"length of lettergreep_words: {len(lettergreep_words)}")
# print(f"length of afterPunctuationMergeFrogMorphedLettergreep: {len(afterPunctuationLemmaPosLettergreep)}")
# print("")
# print(afterPunctuationLemmaPosLettergreep[234])
# print(lettergreep_words[234])
# print(afterPunctuationLemmaPosLettergreep[1234])
# print(lettergreep_words[1234])
# print(afterPunctuationLemmaPosLettergreep[11234])
# print(lettergreep_words[11234])
# print(afterPunctuationLemmaPosLettergreep[500])
# print(lettergreep_words[500])
# print(afterPunctuationLemmaPosLettergreep[546:556])
# print(lettergreep_words[546:556])
# print("FINISHED FINISHED ANALYSIS ANALYSIS")
#
#
# # afterPunctuationMergeMorfessorMorfedLettergreep = convertToMorfessorMorf(lettergreep_words, merge=True, save = True,filename="TrainFiles/afterPunctuationMergeMorfessorMorfedLettergreep.pickle")#change_text_to_morphs(lettergreep_words, frog_merge=True,save = True,filename="TrainFiles/afterPunctuationMergeFrogMorphedLettergreep.pickle")
# # with open('TrainFiles/afterPunctuationMergeMorfessorMorfedLettergreep.pickle', 'rb') as inputfile:
# #     afterPunctuationMergeMorfessorMorfedLettergreep = pickle.load(inputfile)
# #
# # print(afterPunctuationMergeMorfessorMorfedLettergreep[:10])
# # for i in range(len(afterPunctuationMergeMorfessorMorfedLettergreep)):
# #     afterPunctuationMergeMorfessorMorfedLettergreep[i] = afterPunctuationMergeMorfessorMorfedLettergreep[i].replace(" [MERGE] ", " ").replace(" [MERGE]", " ")
# # print(afterPunctuationMergeMorfessorMorfedLettergreep[:10])
# # print("How many with punctuation MorphMerge: ")
# #
# # i=0
# # for word in afterPunctuationMergeMorfessorMorfedLettergreep:
# #     contains = [c for c in word if c in punctuation]
# #     if len(contains) >0:
# #         #print(word)
# #         i+=1
# # print(i)
# # print("with")
# # print(f"length: {len(afterPunctuationMergeMorfessorMorfedLettergreep)}")
# #
# # print("")
# # print("ANALYSIS ANALYSIS")
# # print(f"length of lettergreep_words: {len(lettergreep_words)}")
# # print(f"length of afterPunctuationMergeFrogMorphedLettergreep: {len(afterPunctuationMergeMorfessorMorfedLettergreep)}")
# # print("")
# # print(afterPunctuationMergeMorfessorMorfedLettergreep[234])
# # print(lettergreep_words[234])
# # print(afterPunctuationMergeMorfessorMorfedLettergreep[1234])
# # print(lettergreep_words[1234])
# # print(afterPunctuationMergeMorfessorMorfedLettergreep[11234])
# # print(lettergreep_words[11234])
# # print(afterPunctuationMergeMorfessorMorfedLettergreep[500])
# # print(lettergreep_words[500])
# # print(afterPunctuationMergeMorfessorMorfedLettergreep[546:556])
# # print(lettergreep_words[546:556])
# # print("FINISHED FINISHED ANALYSIS ANALYSIS")
#
# # afterPunctuationMergePolyglotMorfedLettergreep = convertToPolyglotMorf(lettergreep_words, merge=True, save = True,filename="TrainFiles/afterPunctuationMergePolyglotMorfedLettergreep.pickle")#change_text_to_morphs(lettergreep_words, frog_merge=True,save = True,filename="TrainFiles/afterPunctuationMergeFrogMorphedLettergreep.pickle")
# # with open('TrainFiles/afterPunctuationMergePolyglotMorfedLettergreep.pickle', 'rb') as inputfile:
# #     afterPunctuationMergePolyglotMorfedLettergreep = pickle.load(inputfile)
# #
# # print(afterPunctuationMergePolyglotMorfedLettergreep[:10])
# # for i in range(len(afterPunctuationMergePolyglotMorfedLettergreep)):
# #     afterPunctuationMergePolyglotMorfedLettergreep[i] = afterPunctuationMergePolyglotMorfedLettergreep[i].replace(" [MERGE] ", " ").replace(" [MERGE]", " ")
# # print(afterPunctuationMergePolyglotMorfedLettergreep[:10])
# # print("How many with punctuation MorphMerge: ")
# #
# # i=0
# # for word in afterPunctuationMergePolyglotMorfedLettergreep:
# #     contains = [c for c in word if c in punctuation]
# #     if len(contains) >0:
# #         #print(word)
# #         i+=1
# # print(i)
# # print("with")
# # print(f"length: {len(afterPunctuationMergePolyglotMorfedLettergreep)}")
# #
# # print("")
# # print("ANALYSIS ANALYSIS")
# # print(f"length of lettergreep_words: {len(lettergreep_words)}")
# # print(f"length of afterPunctuationMergeFrogMorphedLettergreep: {len(afterPunctuationMergePolyglotMorfedLettergreep)}")
# # print("")
# # print(afterPunctuationMergePolyglotMorfedLettergreep[234])
# # print(lettergreep_words[234])
# # print(afterPunctuationMergePolyglotMorfedLettergreep[1234])
# # print(lettergreep_words[1234])
# # print(afterPunctuationMergePolyglotMorfedLettergreep[11234])
# # print(lettergreep_words[11234])
# # print(afterPunctuationMergePolyglotMorfedLettergreep[500])
# # print(lettergreep_words[500])
# # print(afterPunctuationMergePolyglotMorfedLettergreep[546:556])
# # print(lettergreep_words[546:556])
# # print("FINISHED FINISHED ANALYSIS ANALYSIS")
#
#
#



#------------------------------Script for Frog Lemma Pos Sentiment -----------------------------


# with open('TrainFiles/FrogLemmaPosTrainReviewsSeperate.pickle', 'rb') as inputfile:
#     FrogLemmaPosTrainReviews = pickle.load(inputfile)
#     # for i in range(len(PolyglotMorfTrainReviewsMergeEdited)):
#         # PolyglotMorfTrainReviewsMergeEdited[i] = PolyglotMorfTrainReviewsMergeEdited[i].replace(" [MERGE] "," insertmergetoken ")
#         # PolyglotMorfTrainReviewsMergeEdited[i] = ''.join([c for c in PolyglotMorfTrainReviewsMergeEdited[i] if c not in punctuation])
#         # PolyglotMorfTrainReviewsMergeEdited[i] = PolyglotMorfTrainReviewsMergeEdited[i].replace(" insertmergetoken "," [MERGE] ")
#
# print(FrogLemmaPosTrainReviews[:2])
#
# with open('TrainFiles/FrogLemmaPosTestReviewsSeperate.pickle', 'rb') as inputfile:
#     FrogLemmaPosTestReviews = pickle.load(inputfile)
#     # for i in range(len(PolyglotMorfTestReviewsMergeEdited)):
#
# #print(FrogLemmaPosTestReviews)

#------------------------------Script for Morfessor Sentiment -----------------------------

# with open('TrainFiles/totTrainListShuffled.data', 'rb') as filehandle:
#     # read the data as binary data stream
#     totTrainList = pickle.load(filehandle)
#
# with open('TrainFiles/totTestListShuffled.data', 'rb') as filehandle:
#     # read the data as binary data stream
#     totTestList = pickle.load(filehandle)
#
# # with open('TrainFiles/totTrainListShuffled.pickle', 'wb') as outputfile:
# #     pickle.dump(totTrainList, outputfile)
# #
# # with open('TrainFiles/totTestListShuffled.pickle', 'wb') as outputfile:
# #     pickle.dump(totTestList, outputfile)
#
#
# io = morfessor.MorfessorIO()
#
# model = io.read_binary_model_file('TrainFiles/Officialmodel.bin')
# morfed_sentences = []
# i = 1
# for sentenceToBeProcessed in totTestList:
#     print(i)
#     print("of")
#     print(len(totTestList))
#     i+=1
#     # print(sentenceToBeProcessed)
#     # print("--------------------------------------------------")
#     sentenceToBeProcessed = sentenceToBeProcessed.replace("\n", " ")
#     tokenized_sentence = ucto_tokenize(sentenceToBeProcessed)
#     morfed_sentence = []
#     for w in tokenized_sentence:
#         word_morphemes = model.viterbi_segment(w)
#         word_morphemes = list(filter(None, word_morphemes))
#         #print(word_morphemes)
#         #print(word_morphemes[0])
#         morfed_sentence += intersperse(word_morphemes[0], "[MERGE]")
#     morfed_sentence = ' '.join(morfed_sentence)
#     # print(morfed_sentence)
#     # print("--------------------------------------------------")
#     # print("--------------------------------------------------")
#     # print("--------------------------------------------------")
#     morfed_sentences.append(morfed_sentence)
#
# print(morfed_sentences[:10])
# with open("TrainFiles/MorfessorMorfTestReviewsMergeEdited.pickle", 'wb') as outputfile:
#     pickle.dump(morfed_sentences, outputfile)








#------------------------------Script for MORPH nederlandseZinnen-----------------------------





# from frog import Frog, FrogOptions
# import ucto
#
#
# ## Taken from https://stackoverflow.com/questions/5920643/add-an-item-between-each-item-already-in-the-list
# def intersperse(lst, item):
#     result = [item] * (len(lst) * 2 - 1)
#     result[0::2] = lst
#     return result
#
#
# print("TESTESSTTTT")
# print(intersperse(["mee", "speel", "en", "de"], "__ADD_MERGE__"))
# print(intersperse(["ik"], "__ADD_MERGE__"))
#
#
# def change_text_to_morphs(sentences, frog_merge=False, save=False, filename=None):
#     # sentence list to sentence list in frog morphism form
#     morphSentences = []
#
#     frog = Frog(
#         FrogOptions(tok=True, lemma=True, morph=True, daringmorph=False, mwu=False, chunking=False, ner=False,
#                     parser=False))
#
#     for sentenceNumber in range(0, len(sentences)):
#         print(sentenceNumber)
#         print("of")
#         print(len(sentences))
#         sentenceToBeProcessed = sentences[sentenceNumber]
#         sentenceToBeProcessed = sentenceToBeProcessed.replace("\n", " ")
#         morphSentence = []
#         output = frog.process(sentenceToBeProcessed)
#         for i in range(0, len(output)):
#             morphisms_word = output[i].get("morph")
#             morphisms_word_list = morphisms_word.replace('[', '').split(']')
#             if frog_merge:
#                 morphisms_word_list = list(filter(None, morphisms_word_list))
#                 morphisms_word_list = intersperse(morphisms_word_list, "[MERGE]")
#                 # print(morphisms_word_list)
#             # print("EVET")
#             # print(morphisms_word_list)
#             morphSentence += morphisms_word_list
#         # print("MORPHSENTENCE")
#         # print(morphSentence)
#         # Remove the empty strings
#         morphSentence = list(filter(None, morphSentence))
#         # print("ok")
#         # print(morphSentence)
#         morphSentence = ' '.join(morphSentence)
#         # print("HERE")
#         # print(morphSentence)
#         morphSentences.append(morphSentence)
#
#     if save is True:
#         with open(filename, 'wb') as outputfile:
#             pickle.dump(morphSentences, outputfile)
#     return morphSentences
#
#
# randomEnglishSentence = "hey this is just a tokenized sentence do we need lowerlevel characters or different, who knows let's see if splelling mistakes ge interpreted"
# nederlandseZin = "Hey heb je je aangetekende brief wel degelijk gekregen Ik hoop van wel"
# nederlandseZin2 = "In de Europese Unie spreken ongeveer 25 miljoen mensen Nederlands als eerste taal, en een bijkomende acht miljoen als tweede taal. De Franse Westhoek en de regio rondom de Duitse stad Kleef zijn van oudsher Nederlandstalige gebieden, waar Nederlandse dialecten mogelijk nog gesproken worden door de oudste generaties. Ook in de voormalige kolonie Indonesië kunnen in sommige gebieden de oudste generaties nog Nederlands spreken. Het aantal sprekers van het Nederlands in de Verenigde Staten, Canada en Australië wordt geschat op ruim een half miljoen."
# nederlandseZin3 = "Dit is een korte nederlandse zin die getokenized moet worden"
# nederlandseZin4 = "Wordt deze korte zin optimaal gesegmenteerd? zeker?? Ja hoor..."
# nederlandseZin4 = "Wordt deze korte nederlandse zin optimaal gesegmenteerd"
# nederlandseZin5 = "Het Nederlands is een West-Germaanse taal en de officiële taal van Nederland, Suriname en een van de drie officiële talen van België. Binnen het Koninkrijk der Nederlanden is het Nederlands ook een officiële taal van Aruba, Curaçao en Sint-Maarten. Het Nederlands is de derde meest gesproken Germaanse taal. In de Europese Unie spreken ongeveer 25 miljoen mensen Nederlands als eerste taal, en een bijkomende acht miljoen als tweede taal. De Franse Westhoek en de regio rondom de Duitse stad Kleef zijn van oudsher Nederlandstalige gebieden, waar Nederlandse dialecten mogelijk nog gesproken worden door de oudste generaties. Ook in de voormalige kolonie Indonesië kunnen in sommige gebieden de oudste generaties nog Nederlands spreken. Het aantal sprekers van het Nederlands in de Verenigde Staten, Canada en Australië wordt geschat op ruim een half miljoen. Het Afrikaans, een van de officiële talen van Zuid-Afrika, is een dochtertaal van het Nederlands en beide talen zijn onderling verstaanbaar."
# nederlandseZin6 = "Wanneer het Nederlands als op zichzelf staande taal precies is ontstaan, is onbekend. Het Nederlands in zijn vroegste bekende vorm is het weinig gedocumenteerde Oudnederlands (voor 1170), dat eerst overloopt in het Middelnederlands, ook wel Diets genoemd (1170-1500), en daarna in het Nieuwnederlands. De scheiding tussen de continentale en de kustvarianten van het West-Germaans liep vóór de 5e eeuw dwars door wat nu Nederland en Noordwest-Duitsland heet. De kusttaal (in de wetenschappelijk literatuur Ingveoons genoemd) verspreidde zich aan de hand van Ingveoonse klankverschuivingen in afnemende mate in zuidoostelijke richting. De Friese en Saksische dialecten hebben op het vasteland het meest onder invloed ervan gestaan. In mindere mate hebben ook het West-Vlaams en het Hollands Ingveoonse kenmerken, dialecten die aan de basis hebben gestaan van het huidige Standaardnederlands. In Engeland werd het Angelsaksisch, ook een van de Ingveoonse talen, na de Normandische invasie (1066) sterk geromaniseerd. Alleen in het Fries bleef de kusttaal op het continent bewaard. Door de opeenvolgende Hoogduitse klankverschuivingen ontwikkelde zich tussen de 4e en de 9e eeuw in het continentale West-Germaans een verwijdering tussen het zogenaamde Nederfrankisch en Nedersaksisch aan de ene zijde en het Middelduits en Opperduits aan de andere zijde. Het Nederfrankisch zou uiteindelijk de basis worden van wat nu Nederlands is, terwijl het huidige Duits zijn basis vooral in het Opperduits heeft.[2] De taalscheiding verdiepte zich niet alleen, maar schoof ook geografisch naar het noorden op. Pas in de 16e eeuw begonnen de vele regionale talen in de gebieden waar nu Nederlands wordt gesproken aan hun ontwikkeling tot één standaardtaal. Tot dan toe kende elke regio haar eigen geschreven vorm(en) en daarin weken die in het zuidoosten (Limburg) en noordoosten (van Groningen tot de Achterhoek) het meest af. Zij vertoonden invloeden van de talen van het Hanzegebied en Münsterland en zouden later nauwelijks deelnemen aan de vorming van een algemene Nederlandse standaardtaal. Het economisch en bestuurlijk zwaartepunt in Vlaanderen, Holland en Brabant, met rond de 85% van alle Nederlandstalige inwoners van de Nederlanden, weerspiegelde zich ook in de dominantie van de geschreven varianten uit die gewesten.[3] Deze schrijftalen waren academisch omdat ze vooral op de kanselarijen van vorsten, kloosters en steden en nauwelijks door de ongeletterde bevolking werden gebruikt. Rond 1500 kwam er een streven op gang om een algemene schrijftaal te ontwikkelen die in ruimere gebieden bruikbaar kon zijn door verschillende regionale elementen in zich te verenigen. Dat was ook een behoefte vanuit de centralisering van het bestuur onder het Bourgondische hertogschap dat zijn gezag vanuit Brussel over de gehele Nederlanden wilde uitbreiden, een streven waarin keizer Karel V ten slotte ook zou slagen. In de Reformatie waren het vooral de Bijbelvertalingen en religieuze traktaten waarmee een brede verspreiding werd beoogd, en welke daarom doelbewust in een algemene schrijftaal werden gesteld. Voorlopig bleef het bij pogingen waarin elke auteur zijn eigen streektaal het meeste gewicht gaf. Als benaming voor het Nederlands gelden gedurende Middeleeuwen vooral varianten van Diets/Duuts, het woord Nederlands wordt in 1482 voor het eerst aangetroffen. In de tweede helft van de 16e eeuw komt hier, als synoniem, het woord Nederduytsch bij. Het betreft hier een samenvoeging van Nederlands/Nederlanden en Diets/Duuts (nadat de uu-klank in het Nieuwnederlands naar een ui-klank omboog) en vindt haar oorsprong bij de Rederijkers. Na 1750 neemt het gebruik van Nederduytsch gestaag af, met een duidelijke versnelling na 1815. Met uitzondering van de periode tussen 1651-1700 is Nederlands vanaf 1550 de populairste benaming voor de Nederlandse taal.[4][5][6] De gesproken taal van de hogere standen ging zich pas langzamerhand naar deze nieuwe standaardtaal richten, althans in de noordelijke Nederlanden en het eerst in Holland. Hiermee vond de scheiding in ontwikkeling plaats tussen het Nederlands in Nederland waar men de standaardtaal ook ging spreken, en Vlaanderen waar de hogere standen op het Frans overgingen. De gesproken taal van de lagere standen bleef een gewestelijke of een stedelijke variant totdat de bevolking onder de leerplicht het Nederlands als schrijftaal werd geleerd en zij na enkele generaties die taal ook kon gaan spreken. Hoe langzaam dit proces moest verlopen mag blijken uit de analfabetencijfers, tevens indicaties voor schoolbezoek, die rond 1800 in de noordelijke Nederlanden nog een derde en in Vlaanderen twee derden van de volwassen bevolking omvatten. Om de geschreven Nederlandse standaardtaal tot een dagelijkse omgangstaal te maken moest, met de school als basis, uitbreiding ontstaan van de taalgebruikfuncties. Een doorslaggevende rol speelden daarin de nationaal georganiseerde massamedia en de bovenregionale communicatie ten gevolge van een sterk toenemende bevolkingsmobiliteit."
#
# test_twitterReview_neg = 'heeeeel sleeecht nie aan te zien zeer matig'
#
# nederlandseZin = change_text_to_morphs([nederlandseZin], frog_merge=True)[0]
# nederlandseZin2 = change_text_to_morphs([nederlandseZin2], frog_merge=True)[0]
# nederlandseZin3 = change_text_to_morphs([nederlandseZin3], frog_merge=True)[0]
# nederlandseZin4 = change_text_to_morphs([nederlandseZin4], frog_merge=True)[0]
# nederlandseZin5 = change_text_to_morphs([nederlandseZin5], frog_merge=True)[0]
# nederlandseZin6 = change_text_to_morphs([nederlandseZin6], frog_merge=True)[0]
#
# allNederlandseZinnen = nederlandseZin + "\n\n\n" + nederlandseZin2 + "\n\n\n" + nederlandseZin3 + "\n\n\n" + nederlandseZin4 + "\n\n\n" + nederlandseZin5 + "\n\n\n" + nederlandseZin6
#
#
# print(allNederlandseZinnen)
#
#
# with open("TrainFiles/allNederlandseZinnenFrog.txt", "w") as text_file:
#     text_file.write(allNederlandseZinnen)
#

#------------------------------Script for Polyglot MORF nederlandseZinnen-----------------------------



# randomEnglishSentence = "hey this is just a tokenized sentence do we need lowerlevel characters or different, who knows let's see if splelling mistakes ge interpreted"
# nederlandseZin = "Hey heb je je aangetekende brief wel degelijk gekregen Ik hoop van wel"
# nederlandseZin2 = "In de Europese Unie spreken ongeveer 25 miljoen mensen Nederlands als eerste taal, en een bijkomende acht miljoen als tweede taal. De Franse Westhoek en de regio rondom de Duitse stad Kleef zijn van oudsher Nederlandstalige gebieden, waar Nederlandse dialecten mogelijk nog gesproken worden door de oudste generaties. Ook in de voormalige kolonie Indonesië kunnen in sommige gebieden de oudste generaties nog Nederlands spreken. Het aantal sprekers van het Nederlands in de Verenigde Staten, Canada en Australië wordt geschat op ruim een half miljoen."
# nederlandseZin3 = "Dit is een korte nederlandse zin die getokenized moet worden"
# nederlandseZin4 = "Wordt deze korte zin optimaal gesegmenteerd? zeker?? Ja hoor..."
# nederlandseZin4 = "Wordt deze korte nederlandse zin optimaal gesegmenteerd"
# nederlandseZin5 = "Het Nederlands is een West-Germaanse taal en de officiële taal van Nederland, Suriname en een van de drie officiële talen van België. Binnen het Koninkrijk der Nederlanden is het Nederlands ook een officiële taal van Aruba, Curaçao en Sint-Maarten. Het Nederlands is de derde meest gesproken Germaanse taal. In de Europese Unie spreken ongeveer 25 miljoen mensen Nederlands als eerste taal, en een bijkomende acht miljoen als tweede taal. De Franse Westhoek en de regio rondom de Duitse stad Kleef zijn van oudsher Nederlandstalige gebieden, waar Nederlandse dialecten mogelijk nog gesproken worden door de oudste generaties. Ook in de voormalige kolonie Indonesië kunnen in sommige gebieden de oudste generaties nog Nederlands spreken. Het aantal sprekers van het Nederlands in de Verenigde Staten, Canada en Australië wordt geschat op ruim een half miljoen. Het Afrikaans, een van de officiële talen van Zuid-Afrika, is een dochtertaal van het Nederlands en beide talen zijn onderling verstaanbaar."
# nederlandseZin6 = "Wanneer het Nederlands als op zichzelf staande taal precies is ontstaan, is onbekend. Het Nederlands in zijn vroegste bekende vorm is het weinig gedocumenteerde Oudnederlands (voor 1170), dat eerst overloopt in het Middelnederlands, ook wel Diets genoemd (1170-1500), en daarna in het Nieuwnederlands. De scheiding tussen de continentale en de kustvarianten van het West-Germaans liep vóór de 5e eeuw dwars door wat nu Nederland en Noordwest-Duitsland heet. De kusttaal (in de wetenschappelijk literatuur Ingveoons genoemd) verspreidde zich aan de hand van Ingveoonse klankverschuivingen in afnemende mate in zuidoostelijke richting. De Friese en Saksische dialecten hebben op het vasteland het meest onder invloed ervan gestaan. In mindere mate hebben ook het West-Vlaams en het Hollands Ingveoonse kenmerken, dialecten die aan de basis hebben gestaan van het huidige Standaardnederlands. In Engeland werd het Angelsaksisch, ook een van de Ingveoonse talen, na de Normandische invasie (1066) sterk geromaniseerd. Alleen in het Fries bleef de kusttaal op het continent bewaard. Door de opeenvolgende Hoogduitse klankverschuivingen ontwikkelde zich tussen de 4e en de 9e eeuw in het continentale West-Germaans een verwijdering tussen het zogenaamde Nederfrankisch en Nedersaksisch aan de ene zijde en het Middelduits en Opperduits aan de andere zijde. Het Nederfrankisch zou uiteindelijk de basis worden van wat nu Nederlands is, terwijl het huidige Duits zijn basis vooral in het Opperduits heeft.[2] De taalscheiding verdiepte zich niet alleen, maar schoof ook geografisch naar het noorden op. Pas in de 16e eeuw begonnen de vele regionale talen in de gebieden waar nu Nederlands wordt gesproken aan hun ontwikkeling tot één standaardtaal. Tot dan toe kende elke regio haar eigen geschreven vorm(en) en daarin weken die in het zuidoosten (Limburg) en noordoosten (van Groningen tot de Achterhoek) het meest af. Zij vertoonden invloeden van de talen van het Hanzegebied en Münsterland en zouden later nauwelijks deelnemen aan de vorming van een algemene Nederlandse standaardtaal. Het economisch en bestuurlijk zwaartepunt in Vlaanderen, Holland en Brabant, met rond de 85% van alle Nederlandstalige inwoners van de Nederlanden, weerspiegelde zich ook in de dominantie van de geschreven varianten uit die gewesten.[3] Deze schrijftalen waren academisch omdat ze vooral op de kanselarijen van vorsten, kloosters en steden en nauwelijks door de ongeletterde bevolking werden gebruikt. Rond 1500 kwam er een streven op gang om een algemene schrijftaal te ontwikkelen die in ruimere gebieden bruikbaar kon zijn door verschillende regionale elementen in zich te verenigen. Dat was ook een behoefte vanuit de centralisering van het bestuur onder het Bourgondische hertogschap dat zijn gezag vanuit Brussel over de gehele Nederlanden wilde uitbreiden, een streven waarin keizer Karel V ten slotte ook zou slagen. In de Reformatie waren het vooral de Bijbelvertalingen en religieuze traktaten waarmee een brede verspreiding werd beoogd, en welke daarom doelbewust in een algemene schrijftaal werden gesteld. Voorlopig bleef het bij pogingen waarin elke auteur zijn eigen streektaal het meeste gewicht gaf. Als benaming voor het Nederlands gelden gedurende Middeleeuwen vooral varianten van Diets/Duuts, het woord Nederlands wordt in 1482 voor het eerst aangetroffen. In de tweede helft van de 16e eeuw komt hier, als synoniem, het woord Nederduytsch bij. Het betreft hier een samenvoeging van Nederlands/Nederlanden en Diets/Duuts (nadat de uu-klank in het Nieuwnederlands naar een ui-klank omboog) en vindt haar oorsprong bij de Rederijkers. Na 1750 neemt het gebruik van Nederduytsch gestaag af, met een duidelijke versnelling na 1815. Met uitzondering van de periode tussen 1651-1700 is Nederlands vanaf 1550 de populairste benaming voor de Nederlandse taal.[4][5][6] De gesproken taal van de hogere standen ging zich pas langzamerhand naar deze nieuwe standaardtaal richten, althans in de noordelijke Nederlanden en het eerst in Holland. Hiermee vond de scheiding in ontwikkeling plaats tussen het Nederlands in Nederland waar men de standaardtaal ook ging spreken, en Vlaanderen waar de hogere standen op het Frans overgingen. De gesproken taal van de lagere standen bleef een gewestelijke of een stedelijke variant totdat de bevolking onder de leerplicht het Nederlands als schrijftaal werd geleerd en zij na enkele generaties die taal ook kon gaan spreken. Hoe langzaam dit proces moest verlopen mag blijken uit de analfabetencijfers, tevens indicaties voor schoolbezoek, die rond 1800 in de noordelijke Nederlanden nog een derde en in Vlaanderen twee derden van de volwassen bevolking omvatten. Om de geschreven Nederlandse standaardtaal tot een dagelijkse omgangstaal te maken moest, met de school als basis, uitbreiding ontstaan van de taalgebruikfuncties. Een doorslaggevende rol speelden daarin de nationaal georganiseerde massamedia en de bovenregionale communicatie ten gevolge van een sterk toenemende bevolkingsmobiliteit."
#
# test_twitterReview_neg = 'heeeeel sleeecht nie aan te zien zeer matig'
#
# nederlandseZin = convertToPolyglotMorf([nederlandseZin], merge=True)[0]
# nederlandseZin2 = convertToPolyglotMorf([nederlandseZin2], merge=True)[0]
# nederlandseZin3 = convertToPolyglotMorf([nederlandseZin3], merge=True)[0]
# nederlandseZin4 = convertToPolyglotMorf([nederlandseZin4], merge=True)[0]
# nederlandseZin5 = convertToPolyglotMorf([nederlandseZin5], merge=True)[0]
# nederlandseZin6 = convertToPolyglotMorf([nederlandseZin6], merge=True)[0]
#
# print(nederlandseZin)
#
# allNederlandseZinnen = nederlandseZin + "\n\n\n" + nederlandseZin2 + "\n\n\n" + nederlandseZin3 + "\n\n\n" + nederlandseZin4 + "\n\n\n" + nederlandseZin5 + "\n\n\n" + nederlandseZin6
#
#
# print(allNederlandseZinnen)
#
#
# with open("TrainFiles/allNederlandseZinnenPolyglot.txt", "w") as text_file:
#     text_file.write(allNederlandseZinnen)
# #

#------------------------------Script for Morfessor MORF nederlandseZinnen-----------------------------


#
# randomEnglishSentence = "hey this is just a tokenized sentence do we need lowerlevel characters or different, who knows let's see if splelling mistakes ge interpreted"
# nederlandseZin = "Hey heb je je aangetekende brief wel degelijk gekregen Ik hoop van wel"
# nederlandseZin2 = "In de Europese Unie spreken ongeveer 25 miljoen mensen Nederlands als eerste taal, en een bijkomende acht miljoen als tweede taal. De Franse Westhoek en de regio rondom de Duitse stad Kleef zijn van oudsher Nederlandstalige gebieden, waar Nederlandse dialecten mogelijk nog gesproken worden door de oudste generaties. Ook in de voormalige kolonie Indonesië kunnen in sommige gebieden de oudste generaties nog Nederlands spreken. Het aantal sprekers van het Nederlands in de Verenigde Staten, Canada en Australië wordt geschat op ruim een half miljoen."
# nederlandseZin3 = "Dit is een korte nederlandse zin die getokenized moet worden"
# nederlandseZin4 = "Wordt deze korte zin optimaal gesegmenteerd? zeker?? Ja hoor..."
# nederlandseZin4 = "Wordt deze korte nederlandse zin optimaal gesegmenteerd"
# nederlandseZin5 = "Het Nederlands is een West-Germaanse taal en de officiële taal van Nederland, Suriname en een van de drie officiële talen van België. Binnen het Koninkrijk der Nederlanden is het Nederlands ook een officiële taal van Aruba, Curaçao en Sint-Maarten. Het Nederlands is de derde meest gesproken Germaanse taal. In de Europese Unie spreken ongeveer 25 miljoen mensen Nederlands als eerste taal, en een bijkomende acht miljoen als tweede taal. De Franse Westhoek en de regio rondom de Duitse stad Kleef zijn van oudsher Nederlandstalige gebieden, waar Nederlandse dialecten mogelijk nog gesproken worden door de oudste generaties. Ook in de voormalige kolonie Indonesië kunnen in sommige gebieden de oudste generaties nog Nederlands spreken. Het aantal sprekers van het Nederlands in de Verenigde Staten, Canada en Australië wordt geschat op ruim een half miljoen. Het Afrikaans, een van de officiële talen van Zuid-Afrika, is een dochtertaal van het Nederlands en beide talen zijn onderling verstaanbaar."
# nederlandseZin6 = "Wanneer het Nederlands als op zichzelf staande taal precies is ontstaan, is onbekend. Het Nederlands in zijn vroegste bekende vorm is het weinig gedocumenteerde Oudnederlands (voor 1170), dat eerst overloopt in het Middelnederlands, ook wel Diets genoemd (1170-1500), en daarna in het Nieuwnederlands. De scheiding tussen de continentale en de kustvarianten van het West-Germaans liep vóór de 5e eeuw dwars door wat nu Nederland en Noordwest-Duitsland heet. De kusttaal (in de wetenschappelijk literatuur Ingveoons genoemd) verspreidde zich aan de hand van Ingveoonse klankverschuivingen in afnemende mate in zuidoostelijke richting. De Friese en Saksische dialecten hebben op het vasteland het meest onder invloed ervan gestaan. In mindere mate hebben ook het West-Vlaams en het Hollands Ingveoonse kenmerken, dialecten die aan de basis hebben gestaan van het huidige Standaardnederlands. In Engeland werd het Angelsaksisch, ook een van de Ingveoonse talen, na de Normandische invasie (1066) sterk geromaniseerd. Alleen in het Fries bleef de kusttaal op het continent bewaard. Door de opeenvolgende Hoogduitse klankverschuivingen ontwikkelde zich tussen de 4e en de 9e eeuw in het continentale West-Germaans een verwijdering tussen het zogenaamde Nederfrankisch en Nedersaksisch aan de ene zijde en het Middelduits en Opperduits aan de andere zijde. Het Nederfrankisch zou uiteindelijk de basis worden van wat nu Nederlands is, terwijl het huidige Duits zijn basis vooral in het Opperduits heeft.[2] De taalscheiding verdiepte zich niet alleen, maar schoof ook geografisch naar het noorden op. Pas in de 16e eeuw begonnen de vele regionale talen in de gebieden waar nu Nederlands wordt gesproken aan hun ontwikkeling tot één standaardtaal. Tot dan toe kende elke regio haar eigen geschreven vorm(en) en daarin weken die in het zuidoosten (Limburg) en noordoosten (van Groningen tot de Achterhoek) het meest af. Zij vertoonden invloeden van de talen van het Hanzegebied en Münsterland en zouden later nauwelijks deelnemen aan de vorming van een algemene Nederlandse standaardtaal. Het economisch en bestuurlijk zwaartepunt in Vlaanderen, Holland en Brabant, met rond de 85% van alle Nederlandstalige inwoners van de Nederlanden, weerspiegelde zich ook in de dominantie van de geschreven varianten uit die gewesten.[3] Deze schrijftalen waren academisch omdat ze vooral op de kanselarijen van vorsten, kloosters en steden en nauwelijks door de ongeletterde bevolking werden gebruikt. Rond 1500 kwam er een streven op gang om een algemene schrijftaal te ontwikkelen die in ruimere gebieden bruikbaar kon zijn door verschillende regionale elementen in zich te verenigen. Dat was ook een behoefte vanuit de centralisering van het bestuur onder het Bourgondische hertogschap dat zijn gezag vanuit Brussel over de gehele Nederlanden wilde uitbreiden, een streven waarin keizer Karel V ten slotte ook zou slagen. In de Reformatie waren het vooral de Bijbelvertalingen en religieuze traktaten waarmee een brede verspreiding werd beoogd, en welke daarom doelbewust in een algemene schrijftaal werden gesteld. Voorlopig bleef het bij pogingen waarin elke auteur zijn eigen streektaal het meeste gewicht gaf. Als benaming voor het Nederlands gelden gedurende Middeleeuwen vooral varianten van Diets/Duuts, het woord Nederlands wordt in 1482 voor het eerst aangetroffen. In de tweede helft van de 16e eeuw komt hier, als synoniem, het woord Nederduytsch bij. Het betreft hier een samenvoeging van Nederlands/Nederlanden en Diets/Duuts (nadat de uu-klank in het Nieuwnederlands naar een ui-klank omboog) en vindt haar oorsprong bij de Rederijkers. Na 1750 neemt het gebruik van Nederduytsch gestaag af, met een duidelijke versnelling na 1815. Met uitzondering van de periode tussen 1651-1700 is Nederlands vanaf 1550 de populairste benaming voor de Nederlandse taal.[4][5][6] De gesproken taal van de hogere standen ging zich pas langzamerhand naar deze nieuwe standaardtaal richten, althans in de noordelijke Nederlanden en het eerst in Holland. Hiermee vond de scheiding in ontwikkeling plaats tussen het Nederlands in Nederland waar men de standaardtaal ook ging spreken, en Vlaanderen waar de hogere standen op het Frans overgingen. De gesproken taal van de lagere standen bleef een gewestelijke of een stedelijke variant totdat de bevolking onder de leerplicht het Nederlands als schrijftaal werd geleerd en zij na enkele generaties die taal ook kon gaan spreken. Hoe langzaam dit proces moest verlopen mag blijken uit de analfabetencijfers, tevens indicaties voor schoolbezoek, die rond 1800 in de noordelijke Nederlanden nog een derde en in Vlaanderen twee derden van de volwassen bevolking omvatten. Om de geschreven Nederlandse standaardtaal tot een dagelijkse omgangstaal te maken moest, met de school als basis, uitbreiding ontstaan van de taalgebruikfuncties. Een doorslaggevende rol speelden daarin de nationaal georganiseerde massamedia en de bovenregionale communicatie ten gevolge van een sterk toenemende bevolkingsmobiliteit."
#
# test_twitterReview_neg = 'heeeeel sleeecht nie aan te zien zeer matig'
#
# nederlandseZin = convertToMorfessorMorf([nederlandseZin], merge=True)[0]
# nederlandseZin2 = convertToMorfessorMorf([nederlandseZin2], merge=True)[0]
# nederlandseZin3 = convertToMorfessorMorf([nederlandseZin3], merge=True)[0]
# nederlandseZin4 = convertToMorfessorMorf([nederlandseZin4], merge=True)[0]
# nederlandseZin5 = convertToMorfessorMorf([nederlandseZin5], merge=True)[0]
# nederlandseZin6 = convertToMorfessorMorf([nederlandseZin6], merge=True)[0]
#
# print(nederlandseZin)
#
# allNederlandseZinnen = nederlandseZin + "\n\n\n" + nederlandseZin2 + "\n\n\n" + nederlandseZin3 + "\n\n\n" + nederlandseZin4 + "\n\n\n" + nederlandseZin5 + "\n\n\n" + nederlandseZin6
#
#
# print(allNederlandseZinnen)
#
#
# with open("TrainFiles/allNederlandseZinnenMorfessor.txt", "w") as text_file:
#     text_file.write(allNederlandseZinnen)
# # #

#------------------------------Script for Frog LemmaPos nederlandseZinnen-----------------------------



# randomEnglishSentence = "hey this is just a tokenized sentence do we need lowerlevel characters or different, who knows let's see if splelling mistakes ge interpreted"
# nederlandseZin = "Hey heb je je aangetekende brief wel degelijk gekregen Ik hoop van wel"
# nederlandseZin2 = "In de Europese Unie spreken ongeveer 25 miljoen mensen Nederlands als eerste taal, en een bijkomende acht miljoen als tweede taal. De Franse Westhoek en de regio rondom de Duitse stad Kleef zijn van oudsher Nederlandstalige gebieden, waar Nederlandse dialecten mogelijk nog gesproken worden door de oudste generaties. Ook in de voormalige kolonie Indonesië kunnen in sommige gebieden de oudste generaties nog Nederlands spreken. Het aantal sprekers van het Nederlands in de Verenigde Staten, Canada en Australië wordt geschat op ruim een half miljoen."
# nederlandseZin3 = "Dit is een korte nederlandse zin die getokenized moet worden"
# nederlandseZin4 = "Wordt deze korte zin optimaal gesegmenteerd? zeker?? Ja hoor..."
# nederlandseZin4 = "Wordt deze korte nederlandse zin optimaal gesegmenteerd"
# nederlandseZin5 = "Het Nederlands is een West-Germaanse taal en de officiële taal van Nederland, Suriname en een van de drie officiële talen van België. Binnen het Koninkrijk der Nederlanden is het Nederlands ook een officiële taal van Aruba, Curaçao en Sint-Maarten. Het Nederlands is de derde meest gesproken Germaanse taal. In de Europese Unie spreken ongeveer 25 miljoen mensen Nederlands als eerste taal, en een bijkomende acht miljoen als tweede taal. De Franse Westhoek en de regio rondom de Duitse stad Kleef zijn van oudsher Nederlandstalige gebieden, waar Nederlandse dialecten mogelijk nog gesproken worden door de oudste generaties. Ook in de voormalige kolonie Indonesië kunnen in sommige gebieden de oudste generaties nog Nederlands spreken. Het aantal sprekers van het Nederlands in de Verenigde Staten, Canada en Australië wordt geschat op ruim een half miljoen. Het Afrikaans, een van de officiële talen van Zuid-Afrika, is een dochtertaal van het Nederlands en beide talen zijn onderling verstaanbaar."
# nederlandseZin6 = "Wanneer het Nederlands als op zichzelf staande taal precies is ontstaan, is onbekend. Het Nederlands in zijn vroegste bekende vorm is het weinig gedocumenteerde Oudnederlands (voor 1170), dat eerst overloopt in het Middelnederlands, ook wel Diets genoemd (1170-1500), en daarna in het Nieuwnederlands. De scheiding tussen de continentale en de kustvarianten van het West-Germaans liep vóór de 5e eeuw dwars door wat nu Nederland en Noordwest-Duitsland heet. De kusttaal (in de wetenschappelijk literatuur Ingveoons genoemd) verspreidde zich aan de hand van Ingveoonse klankverschuivingen in afnemende mate in zuidoostelijke richting. De Friese en Saksische dialecten hebben op het vasteland het meest onder invloed ervan gestaan. In mindere mate hebben ook het West-Vlaams en het Hollands Ingveoonse kenmerken, dialecten die aan de basis hebben gestaan van het huidige Standaardnederlands. In Engeland werd het Angelsaksisch, ook een van de Ingveoonse talen, na de Normandische invasie (1066) sterk geromaniseerd. Alleen in het Fries bleef de kusttaal op het continent bewaard. Door de opeenvolgende Hoogduitse klankverschuivingen ontwikkelde zich tussen de 4e en de 9e eeuw in het continentale West-Germaans een verwijdering tussen het zogenaamde Nederfrankisch en Nedersaksisch aan de ene zijde en het Middelduits en Opperduits aan de andere zijde. Het Nederfrankisch zou uiteindelijk de basis worden van wat nu Nederlands is, terwijl het huidige Duits zijn basis vooral in het Opperduits heeft.[2] De taalscheiding verdiepte zich niet alleen, maar schoof ook geografisch naar het noorden op. Pas in de 16e eeuw begonnen de vele regionale talen in de gebieden waar nu Nederlands wordt gesproken aan hun ontwikkeling tot één standaardtaal. Tot dan toe kende elke regio haar eigen geschreven vorm(en) en daarin weken die in het zuidoosten (Limburg) en noordoosten (van Groningen tot de Achterhoek) het meest af. Zij vertoonden invloeden van de talen van het Hanzegebied en Münsterland en zouden later nauwelijks deelnemen aan de vorming van een algemene Nederlandse standaardtaal. Het economisch en bestuurlijk zwaartepunt in Vlaanderen, Holland en Brabant, met rond de 85% van alle Nederlandstalige inwoners van de Nederlanden, weerspiegelde zich ook in de dominantie van de geschreven varianten uit die gewesten.[3] Deze schrijftalen waren academisch omdat ze vooral op de kanselarijen van vorsten, kloosters en steden en nauwelijks door de ongeletterde bevolking werden gebruikt. Rond 1500 kwam er een streven op gang om een algemene schrijftaal te ontwikkelen die in ruimere gebieden bruikbaar kon zijn door verschillende regionale elementen in zich te verenigen. Dat was ook een behoefte vanuit de centralisering van het bestuur onder het Bourgondische hertogschap dat zijn gezag vanuit Brussel over de gehele Nederlanden wilde uitbreiden, een streven waarin keizer Karel V ten slotte ook zou slagen. In de Reformatie waren het vooral de Bijbelvertalingen en religieuze traktaten waarmee een brede verspreiding werd beoogd, en welke daarom doelbewust in een algemene schrijftaal werden gesteld. Voorlopig bleef het bij pogingen waarin elke auteur zijn eigen streektaal het meeste gewicht gaf. Als benaming voor het Nederlands gelden gedurende Middeleeuwen vooral varianten van Diets/Duuts, het woord Nederlands wordt in 1482 voor het eerst aangetroffen. In de tweede helft van de 16e eeuw komt hier, als synoniem, het woord Nederduytsch bij. Het betreft hier een samenvoeging van Nederlands/Nederlanden en Diets/Duuts (nadat de uu-klank in het Nieuwnederlands naar een ui-klank omboog) en vindt haar oorsprong bij de Rederijkers. Na 1750 neemt het gebruik van Nederduytsch gestaag af, met een duidelijke versnelling na 1815. Met uitzondering van de periode tussen 1651-1700 is Nederlands vanaf 1550 de populairste benaming voor de Nederlandse taal.[4][5][6] De gesproken taal van de hogere standen ging zich pas langzamerhand naar deze nieuwe standaardtaal richten, althans in de noordelijke Nederlanden en het eerst in Holland. Hiermee vond de scheiding in ontwikkeling plaats tussen het Nederlands in Nederland waar men de standaardtaal ook ging spreken, en Vlaanderen waar de hogere standen op het Frans overgingen. De gesproken taal van de lagere standen bleef een gewestelijke of een stedelijke variant totdat de bevolking onder de leerplicht het Nederlands als schrijftaal werd geleerd en zij na enkele generaties die taal ook kon gaan spreken. Hoe langzaam dit proces moest verlopen mag blijken uit de analfabetencijfers, tevens indicaties voor schoolbezoek, die rond 1800 in de noordelijke Nederlanden nog een derde en in Vlaanderen twee derden van de volwassen bevolking omvatten. Om de geschreven Nederlandse standaardtaal tot een dagelijkse omgangstaal te maken moest, met de school als basis, uitbreiding ontstaan van de taalgebruikfuncties. Een doorslaggevende rol speelden daarin de nationaal georganiseerde massamedia en de bovenregionale communicatie ten gevolge van een sterk toenemende bevolkingsmobiliteit."
#
# test_twitterReview_neg = 'heeeeel sleeecht nie aan te zien zeer matig'
#
# nederlandseZin = change_text_to_lemma_POS([nederlandseZin])[0]
# nederlandseZin2 = change_text_to_lemma_POS([nederlandseZin2])[0]
# nederlandseZin3 = change_text_to_lemma_POS([nederlandseZin3])[0]
# nederlandseZin4 = change_text_to_lemma_POS([nederlandseZin4])[0]
# nederlandseZin5 = change_text_to_lemma_POS([nederlandseZin5])[0]
# nederlandseZin6 = change_text_to_lemma_POS([nederlandseZin6])[0]
#
# print(nederlandseZin)
#
# allNederlandseZinnen = nederlandseZin + "\n\n\n" + nederlandseZin2 + "\n\n\n" + nederlandseZin3 + "\n\n\n" + nederlandseZin4 + "\n\n\n" + nederlandseZin5 + "\n\n\n" + nederlandseZin6
#
#
# print(allNederlandseZinnen)
#
#
# with open("TrainFiles/allNederlandseZinnenLemmaPos.txt", "w") as text_file:
#     text_file.write(allNederlandseZinnen)
# # #


#------------------------------Script for DieDat Processing -----------------------------

with open('TrainFiles/train.sentences', 'r') as inputfile:
    diedat_train_sentences = inputfile.readlines()

with open('TrainFiles/train.labels', 'r') as inputfile:
    diedat_train_labels = inputfile.readlines()

with open('TrainFiles/dev.labels', 'r') as inputfile:
    diedat_dev_labels = inputfile.readlines()

with open('TrainFiles/dev.sentences', 'r') as inputfile:
    diedat_dev_sentences = inputfile.readlines()

#print(diedat_train_sentences)

diedat_train_sentences_changed_mask = diedat_train_sentences
for i in range(len(diedat_train_sentences)):
    diedat_train_sentences_changed_mask[i] = diedat_train_sentences[i].replace("[MASK]", "93434343402020219293949001019298")

diedat_dev_sentences_changed_mask = diedat_dev_sentences
for i in range(len(diedat_dev_sentences)):
    diedat_dev_sentences_changed_mask[i] = diedat_dev_sentences[i].replace("[MASK]", "93434343402020219293949001019298")

print(diedat_train_sentences_changed_mask[:5])
print(diedat_dev_sentences_changed_mask[:5])


# convertedTrainToFrog = change_text_to_morphs(diedat_train_sentences, frog_merge=True, save = True, filename="TrainFiles/train_frogpiece.pickle")
# convertedDevToFrog = change_text_to_morphs(diedat_dev_sentences, frog_merge=True, save = True, filename="TrainFiles/dev_frogpiece.pickle")
#
# convertedTrainToLemmaPos = change_text_to_lemma_POS(diedat_train_sentences, save=True, filename="TrainFiles/train_lemmapospiece.pickle")
# convertedDevToLemmaPos = change_text_to_lemma_POS(diedat_dev_sentences, save=True, filename="TrainFiles/dev_lemmapospiece.pickle")


with open('TrainFiles/train_frogpiece.pickle', 'rb') as filehandle:
    # read the data as binary data stream
    train_frogpiece = pickle.load(filehandle)

with open('TrainFiles/dev_frogpiece.pickle', 'rb') as filehandle:
    # read the data as binary data stream
    dev_frogpiece = pickle.load(filehandle)

with open('TrainFiles/train_lemmapospiece.pickle', 'rb') as filehandle:
    # read the data as binary data stream
    train_lemmapospiece = pickle.load(filehandle)

with open('TrainFiles/dev_lemmapospiece.pickle', 'rb') as filehandle:
    # read the data as binary data stream
    dev_lemmapospiece = pickle.load(filehandle)

print("")
print("STARTER")
print("")
print("")
print("")
print("")
print("")
print(train_frogpiece[:10])
print("")
print("SPLITTER")
print("")
print(dev_frogpiece[:10])
print("")
print("SPLITTER")
print("")
print(train_lemmapospiece[:10])
print("")
print("SPLITTER")
print("")
print(dev_lemmapospiece[:10])

#------------------------------ Script for MORPHISM CHECKING -----------------------------

# Open = open('TrainFiles/morphismsBetter.txt')
# CELEX_Morphisms_text = Open.readlines()
# #CELEX_Morphisms_text = CELEX_Morphisms_text.pop(0)
#
# #print(CELEX_Morphisms_text[40:45])
# # remove all that are not equal to word in combination (so mistakes removed + not basic form), remove all with multiple occurences
#
# amount = 1
# total_amount = len(CELEX_Morphisms_text)
#
# splits = []
# words = []
# for line in CELEX_Morphisms_text:
#
#     print(amount)
#     print("of")
#     print(total_amount)
#     amount+=1
#
#     line = line.replace("(","").replace(")","")
#     splitted = line.split('\\')
#     first = splitted[0]
#     second = splitted[1].strip()
#
#
#     first = first.split(",")
#     # print("")
#     # print(f"first: {first}")
#     # print(f"second: {second}")
#     # print(len(second))
#     total_first = ""
#     for morpheme in first:
#         total_first += morpheme
#
#     total_first = total_first.strip()
#
#     # print(f"total_first: {total_first}")
#     #print(len(total_first))
#
#     if total_first == second and words.count(second)==0:
#         splits.append(first)
#         words.append(second)
#
# with open("TrainFiles/words.pickle", 'wb') as outputfile:
#     pickle.dump(words, outputfile)
#
# with open("TrainFiles/splits.pickle", 'wb') as outputfile:
#     pickle.dump(splits, outputfile)



# !!!
#
# with open('TrainFiles/words.pickle', 'rb') as filehandle:
#     # read the data as binary data stream
#     words = pickle.load(filehandle)
#
# with open('TrainFiles/splits.pickle', 'rb') as filehandle:
#     # read the data as binary data stream
#     splits = pickle.load(filehandle)
#
# print(splits[:10])
# print(words[:10])
#
# print("")
# print(splits[137])
# print(words[137])
# print(splits[1370])
# print(words[1370])
# print(splits[13767])
# print(words[13767])
# print(splits[9898])
# print(words[9898])
# print(splits[53400])
# print(words[53400])
# print(splits[234])
# print(words[234])
# print(splits[1])
# print(words[1])
#
# print(f"length of splits is {len(splits)}")
# print(f"length of splits is {len(words)}")
#
#     #print(repr(line))
#     #print(splitted[0])
#     #print(splitted[1])
#
#
# #morfed_totTrainList = morfed_totTrainList.split('*%')
#
# convPolyMorph = convertToPolyglotMorf(words,save=True,filename="TrainFiles/wordsPolyglotted.pickle")
#
# print("")
# print("HERE POLY")
# print("")
# print(convPolyMorph[:10])
#
# print(words[:10])
# print(splits[:10])
#
# total = len(words)
# total_correct = 0
# for i in range(len(convPolyMorph)):
#     print(convPolyMorph[i].split(" "))
#     print(splits[i])
#     if convPolyMorph[i].split(" ") == splits[i]:
#         print("TRUE")
#         #print(convPolyMorph[i].split(" "))
#         #print(splits[i])
#         total_correct += 1
#     else:
#         print("FALSE")
#
# accuracy = total_correct / total
#
# print(f"total: {total}")
# print(f"total correct: {total_correct}")
# print(f"accuracy of polyglot on test_set: {accuracy*100} %")
#
#
#
#
#
#
#
#
# convMorfessorMorph = convertToMorfessorMorf(words,save=True,filename="TrainFiles/wordsMorfessored.pickle")

#
# print("")
# print("HERE MORFESSOR")
# print("")
# print(convMorfessorMorph[:10])
#
# print(words[:10])
# print(splits[:10])
#
# total = len(words)
# total_correct = 0
# for i in range(len(convMorfessorMorph)):
#     print(convMorfessorMorph[i].split(" "))
#     print(splits[i])
#     if convMorfessorMorph[i].split(" ") == splits[i]:
#         print("TRUE")
#         #print(convPolyMorph[i].split(" "))
#         #print(splits[i])
#         total_correct += 1
#     else:
#         print("FALSE")
#
# accuracy = total_correct / total
#
# print(f"total: {total}")
# print(f"total correct: {total_correct}")
# print(f"accuracy of morfessor on test_set: {accuracy*100} %")
#



convFrogMorph = change_text_to_morphs(words,save=True,filename="TrainFiles/wordsFrogged.pickle")
# print(convFrogMorph)
#
# print("")
# print("HERE MORFESSOR")
# print("")
# print(convFrogMorph[:10])
#
# print(words[:10])
# print(splits[:10])
#
# total = len(words)
# total_correct = 0
# for i in range(len(convFrogMorph)):
#     print(convFrogMorph[i].split(" "))
#     print(splits[i])
#     if convFrogMorph[i].split(" ") == splits[i]:
#         print("TRUE")
#         #print(convPolyMorph[i].split(" "))
#         #print(splits[i])
#         total_correct += 1
#     else:
#         print("FALSE")
#
# accuracy = total_correct / total
#
# print(f"total: {total}")
# print(f"total correct: {total_correct}")
# print(f"accuracy of Frog on test_set: {accuracy*100} %")










#total: 71258
#total correct: 31037
#accuracy of polyglot on test_set: 43.555811277330264 %


# total: 71258
# total correct: 29114
# accuracy of polyglot PIECE on test_set: 40.85716691459205 %



#total: 71258
#total correct: 42304
#accuracy of morfessor on test_set: 59.36736927783547 %

# total: 71258
# total correct: 37559
# accuracy of morfessor PIECE on test_set: 52.70846782115692 %



# total: 71258
# total correct: 21902
# accuracy of word PIECE on test_set: 30.73619804092172 %

