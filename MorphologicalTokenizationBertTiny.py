import pickle
import numpy as np
from string import punctuation
import pandas as pd
import matplotlib.pyplot as plt
from HuggingfaceTokenization.OverrideClasses import BertWordPieceTokenizerWhitespacePreTokenizer
from transformersAdjusted import BertTokenizer

import datetime
import time


#------------------------------------------------------------

fixed_len = 128
include_loading_files = True
include_vocabulary_training = False
include_review_encoding = True


include_morph_post_review = False
include_polyglot_morf_post_review = False
include_morfessor_morf_post_review = False
include_poslemma_post_review = False
include_morph_Hashtag_post_review = False

TEST_MORPHOLOGY = False

vocabulary_file_to_use_in_transformers = "VocabularyTokenizer/EUROPARLvocabFROGLEMMAPOS.txt"  #"VocabularyTokenizer/vocabFrogWhitePretokenizer30000.txt" if you want to produce new one in Tokenizer and directly use it


#------------------------------------------------------------

def format_time(elapsed):
  '''
  Takes a time in seconds and returns a string hh:mm:ss
  '''
  # Round to the nearest second.
  elapsed_rounded = int(round((elapsed)))

  # Format as hh:mm:ss
  return str(datetime.timedelta(seconds=elapsed_rounded))

#-------------------------------------------------------------








if include_loading_files == True:

    print("LOADING FILES...")


###################




    # with open('TrainFiles/FrogMorphedMergedEuroparl700k.pickle', 'rb') as inputfile:
    #     FrogMorphedMergedEuroparl700k = pickle.load(inputfile)
    #
    # with open('TrainFiles/FrogMorphedMergedEuroparl700k_1400k.pickle', 'rb') as inputfile:
    #     FrogMorphedMergedEuroparl700k_1400k = pickle.load(inputfile)
    #
    # with open('TrainFiles/FrogMorphedMergedEuroparl1400k_2000k.pickle', 'rb') as inputfile:
    #     FrogMorphedMergedEuroparl1400k_2000k = pickle.load(inputfile)
    #
    #
    # FrogMorphedMergedEuroparlTotal_NonMergeSymbol = FrogMorphedMergedEuroparl700k + FrogMorphedMergedEuroparl700k_1400k + FrogMorphedMergedEuroparl1400k_2000k
    #
    # FrogMorphedMergedEuroparlTotal = FrogMorphedMergedEuroparlTotal_NonMergeSymbol
    # for i in range(len(FrogMorphedMergedEuroparlTotal_NonMergeSymbol)):
    #     FrogMorphedMergedEuroparlTotal[i] = FrogMorphedMergedEuroparlTotal_NonMergeSymbol[i].replace(" __add_merge__ ", " [MERGE] ")
    #     FrogMorphedMergedEuroparlTotal[i] = FrogMorphedMergedEuroparlTotal[i].replace(" [MERGE] ", " ##")

    # with open('TrainFiles/FrogMorphedMergedOscar300k.pickle', 'rb') as inputfile:
    #     FrogMorphedMergedOscar300k = pickle.load(inputfile)
    #
    # with open('TrainFiles/FrogMorphedMergedOscar300k_600k.pickle', 'rb') as inputfile:
    #    FrogMorphedMergedOscar300k_600k = pickle.load(inputfile)
    #
    # with open('TrainFiles/FrogMorphedMergedOscar600k_800k.pickle', 'rb') as inputfile:
    #     FrogMorphedMergedOscar600k_800k = pickle.load(inputfile)
    #
    # with open('TrainFiles/FrogMorphedMergedOscar800k_900k.pickle', 'rb') as inputfile:
    #     FrogMorphedMergedOscar800k_900k = pickle.load(inputfile)
    #
    # with open('TrainFiles/FrogMorphedMergedOscar900k_1200k.pickle', 'rb') as inputfile:
    #     FrogMorphedMergedOscar900k_1200k = pickle.load(inputfile)
    #
    # with open('TrainFiles/FrogMorphedMergedOscar1200k_1400k.pickle', 'rb') as inputfile:
    #     FrogMorphedMergedOscar1200k_1400k = pickle.load(inputfile)
    #
    # with open('TrainFiles/FrogMorphedMergedOscar1400k_1500k.pickle', 'rb') as inputfile:
    #     FrogMorphedMergedOscar1400k_1500k = pickle.load(inputfile)
    #
    # with open('TrainFiles/FrogMorphedMergedOscar1500k_1800k.pickle', 'rb') as inputfile:
    #     FrogMorphedMergedOscar1500k_1800k = pickle.load(inputfile)
    # #
    # with open('TrainFiles/FrogMorphedMergedOscar1800k_2000k.pickle', 'rb') as inputfile:
    #     FrogMorphedMergedOscar1800k_2000k = pickle.load(inputfile)
    # # #
# #    FrogMorphedMergedOscarTotal_NonMergeSymbol = FrogMorphedMergedOscar300k + FrogMorphedMergedOscar300k_600k + FrogMorphedMergedOscar600k_800k + FrogMorphedMergedOscar800k_900k + FrogMorphedMergedOscar900k_1200k# + FrogMorphedMergedOscar1200k_1400k + FrogMorphedMergedOscar1400k_1500k + FrogMorphedMergedOscar1500k_1800k + FrogMorphedMergedOscar1800k_2000k
#     FrogMorphedMergedOscarTotal_NonMergeSymbol = FrogMorphedMergedOscar900k_1200k + FrogMorphedMergedOscar1200k_1400k + FrogMorphedMergedOscar1400k_1500k + FrogMorphedMergedOscar1500k_1800k + FrogMorphedMergedOscar1800k_2000k


    # FrogMorphedMergedOscarTotal = FrogMorphedMergedOscarTotal_NonMergeSymbol
    # for i in range(len(FrogMorphedMergedOscarTotal_NonMergeSymbol)):
    #     FrogMorphedMergedOscarTotal[i] = FrogMorphedMergedOscarTotal_NonMergeSymbol[i].replace(" __add_merge__ ", " [MERGE] ")
    #     FrogMorphedMergedOscarTotal[i] = FrogMorphedMergedOscarTotal[i].replace(" [MERGE] ", " ##")

    # for sentence in FrogMorphedMergedOscarTotal:
    #     if len(sentence.split(" ")) > 100000:
    #        print(sentence)
    #        i+=1
    # print("how many too long oscar")
    # print(i)
    #
    # i = 0
    # for sentence in FrogMorphedMergedEuroparlTotal:
    #     if len(sentence.split(" ")) > 1000:
    #        i+=1
    #        print(sentence)
    # print("how many too long europarl")
    # print(i)

    # LemmaPosFitData = FrogLemmaPosEuroparlTotal[:1000000] + FrogLemmaPosOscarTotal[:300000]

    #
    # print("")
    # print("LOCATE")
    # print(FrogMorphedMergedOscarTotal[:10])
    # print("")

    # SOLELY LEMMAPOS#

    # with open('TrainFiles/FrogLemmaPosEuroparl700k.pickle', 'rb') as inputfile:
    #     FrogLemmaPosEuroparl700k = pickle.load(inputfile)
    #
    # with open('TrainFiles/FrogLemmaPosEuroparl700k_1400k.pickle', 'rb') as inputfile:
    #     FrogLemmaPosEuroparl700k_1400k = pickle.load(inputfile)
    #
    # with open('TrainFiles/FrogLemmaPosEuroparl1400k_2000k.pickle', 'rb') as inputfile:
    #     FrogLemmaPosEuroparl1400k_2000k = pickle.load(inputfile)

    # with open('TrainFiles/FrogLemmaPosOscar100k.pickle', 'rb') as inputfile:
    #    FrogLemmaPosOscar100k = pickle.load(inputfile)
    #
    # with open('TrainFiles/FrogLemmaPosOscar100k_400k.pickle', 'rb') as inputfile:
    #    FrogLemmaPosOscar100k_400k = pickle.load(inputfile)
    #
    # with open('TrainFiles/FrogLemmaPosOscar400k_600k.pickle', 'rb') as inputfile:
    #    FrogLemmaPosOscar400k_600k = pickle.load(inputfile)
#
    # with open('TrainFiles/FrogLemmaPosOscar600k_1000k.pickle', 'rb') as inputfile:
    #    FrogLemmaPosOscar600k_1000k = pickle.load(inputfile)

    # with open('TrainFiles/FrogLemmaPosOscar1000k_1300k.pickle', 'rb') as inputfile:
    #    FrogLemmaPosOscar1000k_1300k = pickle.load(inputfile)
    #
    # with open('TrainFiles/FrogLemmaPosOscar1300k_1400k.pickle', 'rb') as inputfile:
    #    FrogLemmaPosOscar1300k_1400k = pickle.load(inputfile)
    #
    # with open('TrainFiles/FrogLemmaPosOscar1400k_1700k.pickle', 'rb') as inputfile:
    #    FrogLemmaPosOscar1400k_1700k = pickle.load(inputfile)

    with open('TrainFiles/FrogLemmaPosOscar1700k_2000k.pickle', 'rb') as inputfile:
       FrogLemmaPosOscar1700k_2000k = pickle.load(inputfile)
#
#
#     FrogLemmaPosEuroparlTotal = FrogLemmaPosEuroparl700k + FrogLemmaPosEuroparl700k_1400k + FrogLemmaPosEuroparl1400k_2000k
#
    FrogLemmaPosOscarTotal = FrogLemmaPosOscar1700k_2000k
    # FrogLemmaPosEuroparlTotal = FrogLemmaPosEuroparlTotal[1000000:]
    # FrogLemmaPosOscarTotal = FrogLemmaPosOscar1000k_13
# #
#     for i in range(len(FrogLemmaPosEuroparlTotal)):
#         FrogLemmaPosEuroparlTotal[i] = FrogLemmaPosEuroparlTotal[i].replace("**<", "<").replace(">**", ">")
#
    for i in range(len(FrogLemmaPosOscarTotal)):
        FrogLemmaPosOscarTotal[i] = FrogLemmaPosOscarTotal[i].replace("**<", "<").replace(">**", ">")
#
#         ###
#
#
# with open("TrainFiles/FrogLemmaPosEuroparlTokenizerTrainText500k.txt") as f:
#     FrogLemmaPosEuroparlTokenizerTrainText500k = f.readlines()
#
# with open("TrainFiles/FrogLemmaPosEuroparlTokenizerTrainText1000k.txt") as f:
#     FrogLemmaPosEuroparlTokenizerTrainText1000k = f.readlines()
#
# with open("TrainFiles/FrogLemmaPosEuroparlTokenizerTrainText1500k.txt") as f:
#     FrogLemmaPosEuroparlTokenizerTrainText1500k = f.readlines()
#
# with open("TrainFiles/FrogLemmaPosEuroparlTokenizerTrainText2000k.txt") as f:
#     FrogLemmaPosEuroparlTokenizerTrainText2000k = f.readlines()
#
# with open("TrainFiles/FrogLemmaPosOscarTokenizerTrainText100k.txt") as f:
#     FrogLemmaPosOscarTokenizerTrainText100k = f.readlines()
#
# with open("TrainFiles/FrogLemmaPosOscarTokenizerTrainText200k.txt") as f:
#     FrogLemmaPosOscarTokenizerTrainText200k = f.readlines()
#
# with open("TrainFiles/FrogLemmaPosOscarTokenizerTrainText300k.txt") as f:
#     FrogLemmaPosOscarTokenizerTrainText300k = f.readlines()
#
# with open("TrainFiles/FrogLemmaPosOscarTokenizerTrainText400k.txt") as f:
#     FrogLemmaPosOscarTokenizerTrainText400k = f.readlines()
#
# with open("TrainFiles/FrogLemmaPosOscarTokenizerTrainText500k.txt") as f:
#     FrogLemmaPosOscarTokenizerTrainText500k = f.readlines()
#
#
# allEuroparlText = FrogLemmaPosEuroparlTokenizerTrainText500k + FrogLemmaPosEuroparlTokenizerTrainText1000k + FrogLemmaPosEuroparlTokenizerTrainText1500k + FrogLemmaPosEuroparlTokenizerTrainText2000k
# allOscarText = FrogLemmaPosOscarTokenizerTrainText100k + FrogLemmaPosOscarTokenizerTrainText200k + FrogLemmaPosOscarTokenizerTrainText300k + FrogLemmaPosOscarTokenizerTrainText400k + FrogLemmaPosOscarTokenizerTrainText500k
#
# allText = allEuroparlText + allOscarText

#allTextText = " ".join(allText)
#allTextTextWords = allTextText.split(" ")

# print("DONE")
# print("DONE")
# print("DONE")
# print(allTextTextWords[:10])
# print("DONE")
# print("DONE")
# print("DONE")

# amount_of_POSTAGS = 0
# all_POSTAGS = []
# count = 1
# for line in allText:
#     if count % 10000 == 0:
#         print(count)
#         print(f"of {len(allText)}")
#     count += 1
#     lineSplitted = line.split(" ")
#     for i in lineSplitted:
#         if i[0] == "<" and i[-1] == ">" and i not in all_POSTAGS:
#             print(i)
#             amount_of_POSTAGS += 1
#             all_POSTAGS.append(i)
#
# print(amount_of_POSTAGS)
# print(all_POSTAGS)
#
# print("STARTSTARTSTARTSTARTSTART")
# for i in all_POSTAGS:
#     print(i)
# print("ENDENDENDENDENDENDENDEND")
# print("")
# print(f"AMOUNT OF DIFFERENT POSTAGS INSIDE: {amount_of_POSTAGS}")
# print("")


# with open("TrainFiles/allPosTagsLemmaPosEuroparlAndOscar500kText.txt") as f:
#     allPosTagsLemmaPosEuroparlAndOscar500kText = f.readlines()
#
# allPosTagsLemmaPosEuroparlAndOscar500kTextStripped = []
# for postag in allPosTagsLemmaPosEuroparlAndOscar500kText:
#    allPosTagsLemmaPosEuroparlAndOscar500kTextStripped.append(postag.rstrip('\n'))
#
# if  allPosTagsLemmaPosEuroparlAndOscar500kTextStripped.count('<LID><bep><stan><evon>') > 0 :
#     print("YES dIDIDIIIT IT")
# else:
#     print("NO HERGERZHE")
# #
# # for i in allPosTagsLemmaPosEuroparlAndOscar500kText:
# #     print(allPosTagsLemmaPosEuroparlAndOscar500kText)
# #     if "<LID><bep><stan><evon>" == i:
# #         print("YESSSSSSSSSS")
# #     else:
# #         print("NO")
#
#
#
# print(allPosTagsLemmaPosEuroparlAndOscar500kTextStripped[:10])
# print(len(allPosTagsLemmaPosEuroparlAndOscar500kTextStripped))
# print(len(allPosTagsLemmaPosEuroparlAndOscar500kTextStripped))
# print(len(allPosTagsLemmaPosEuroparlAndOscar500kTextStripped))
# print(len(allPosTagsLemmaPosEuroparlAndOscar500kTextStripped))
#
#
#
#
# with open("VocabularyTokenizer/EUROPARLvocabFROGLEMMAPOS.txt") as f:
#     eurolines = f.readlines()
#
# with open("VocabularyTokenizer/EUROPARLAND500KOSCARvocabFROGLEMMAPOS.txt") as f:
#     euroosclines = f.readlines()
#
# euro = []
# for sentence in eurolines:
#    euro.append(sentence.rstrip('\n'))
#
# euroosc = []
# for sentence in euroosclines:
#    euroosc.append(sentence.rstrip('\n'))
#
# print(euro[:10])
# print("")
# print("SPLIT")
# print("")
# print(euroosc[:10])
#
# how_many = 0
# for postag in allPosTagsLemmaPosEuroparlAndOscar500kTextStripped:
#     if postag.lower() in euro:
#         #print(postag.lower())
#         how_many +=1
#
# print("")
# print(f"AMOUNT OF POSTAGS INSIDE Vocabulary of Europarl: {how_many}")
# print("")
#
# how_many = 0
# for postag in allPosTagsLemmaPosEuroparlAndOscar500kTextStripped:
#     if postag.lower() in euroosc:
#         #print(postag.lower())
#         how_many +=1
#
# print("")
# print(f"AMOUNT OF POSTAGS INSIDE Vocabulary of Europarl + 500k Oscar: {how_many}")
# print("")
#
# how_many = 0
# for voc in euro:
#     if "<" in voc or ">" in voc:
#         how_many += 1
#
# print("")
# print(f"AMOUNT OF < or > containing vocs INSIDE Vocabulary of Europarl: {how_many}")
# print("")
#
# how_many = 0
# for voc in euroosc:
#     if "<" in voc or ">" in voc:
#         how_many+=1
#
# print("")
# print(f"AMOUNT OF < or > containing vocs INSIDE Vocabulary of Europarl + 500k Oscar: {how_many}")
# print("")



# FrogLemmaPosOscarTotal = FrogLemmaPosOscarTotal[400000:500000]
#
# print("")
# print("LOCATE")
# print(FrogLemmaPosOscarTotal[:2])
# print("")
#
#
# FrogLemmaPosOscarTokenizerTrainText= '\n'.join(FrogLemmaPosOscarTotal)
# print(FrogLemmaPosOscarTokenizerTrainText[:200])
#
# with open("TrainFiles/FrogLemmaPosOscarTokenizerTrainText500k.txt", "w") as text_file:
#     text_file.write(FrogLemmaPosOscarTokenizerTrainText)




# FrogMorphedMergedEuroparlTokenizerTrain = FrogMorphedMergedEuroparlTotal
# for i in range (len(FrogMorphedMergedEuroparlTotal)):
#     FrogMorphedMergedEuroparlTokenizerTrain[i] = FrogMorphedMergedEuroparlTokenizerTrain[i].replace(" [MERGE] ", " ##")

# print("")
# print("LOCATE")
# print(FrogMorphedMergedEuroparlTokenizerTrain[:10])
# print("")
#
# FrogMorphedMergedEuroparlTokenizerTrainText= '\n'.join(FrogMorphedMergedEuroparlTokenizerTrain)
# print(FrogMorphedMergedEuroparlTokenizerTrainText[:2000])
#
# with open("TrainFiles/FrogMorphedMergedHashtagEuroparlTokenizerTrainText.txt", "w") as text_file:
#     text_file.write(FrogMorphedMergedEuroparlTokenizerTrainText)

####




#FOR Polyglot (no merge for tokenizer ofcourse)

# PolyglotMorfTrainReviewsWithoutMerge = PolyglotMorfTrainReviewsMergeEdited
# for i in range(len(PolyglotMorfTrainReviewsWithoutMerge)):
#     PolyglotMorfTrainReviewsWithoutMerge[i] = PolyglotMorfTrainReviewsWithoutMerge[i].replace(" [MERGE] ", " ")
# allPolyglotMorfTrainText= '\n'.join(PolyglotMorfTrainReviewsWithoutMerge)
# print(allPolyglotMorfTrainText[:2000])
#
# with open("TrainFiles/allPolyglotMorfTrainText.txt", "w") as text_file:
#     text_file.write(allPolyglotMorfTrainText)

#FOR Morfessor (no merge for tokenizer ofcourse)

# MorfessorMorfTrainReviewsWithoutMerge = MorfessorMorfTrainReviewsMergeEdited
# for i in range(len(MorfessorMorfTrainReviewsWithoutMerge)):
#     MorfessorMorfTrainReviewsWithoutMerge[i] = MorfessorMorfTrainReviewsWithoutMerge[i].replace(" [MERGE] ", " ")
# allMorfessorMorfTrainText= '\n'.join(MorfessorMorfTrainReviewsWithoutMerge)
# print(allMorfessorMorfTrainText[:2000])
#
# with open("TrainFiles/allMorfessorMorfTrainText.txt", "w") as text_file:
#     text_file.write(allMorfessorMorfTrainText)

# FOR LemmaPos


# allFrogLemmaPosTrainText = '\n'.join(FrogLemmaPosTrainReviews)
# print(allFrogLemmaPosTrainText[:2000])
#
# with open("TrainFiles/allFrogLemmaPosTrainText.txt", "w") as text_file:
#     text_file.write(allFrogLemmaPosTrainText)

#---------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------- HUGGING FACE TOKENIZERS -----------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------







if include_vocabulary_training is True:

    print("TRAINING TOKENIZER VOCABULARY...")

    # Initialize BertWordPiece tokenizer

    #   - does Bert normalization             * removing any control characters and replacing all whitespaces by the classic one.
    #                                         * handle chinese chars by putting spaces around them.
    #                                         * strip all accents.
    #   - does Bert pretokenization           * This pre-tokenizer splits tokens on spaces, and also on punctuation. Each occurence of a punctuation character will be treated separately.
    #   - does Bert postprocessing            * This post-processor takes care of adding the special tokens needed by a Bert model: CLS and SEP

    # NOTE: pretokenization and postprocessing don't matter here because I use only the tokenizer to construct a vocabulary. NO, PRETOK MATTERS HERE TOO

    tokenizer = BertWordPieceTokenizerWhitespacePreTokenizer(
        lowercase=True,
        wordpieces_prefix = "", #other: ""
        handle_chinese_chars = False
    )

    #!!!!!!!!!!!!!!!!!!!!! watch out MERGE (for lemmapos and morphHashtag turned off)
    # And then train
    tokenizer.train(
        ["TrainFiles/FrogLemmaPosEuroparlTokenizerTrainText500k.txt", "TrainFiles/FrogLemmaPosEuroparlTokenizerTrainText1000k.txt", "TrainFiles/FrogLemmaPosEuroparlTokenizerTrainText1500k.txt", "TrainFiles/FrogLemmaPosEuroparlTokenizerTrainText2000k.txt", "TrainFiles/FrogLemmaPosOscarTokenizerTrainText100k.txt", "TrainFiles/FrogLemmaPosOscarTokenizerTrainText200k.txt", "TrainFiles/FrogLemmaPosOscarTokenizerTrainText300k.txt", "TrainFiles/FrogLemmaPosOscarTokenizerTrainText400k.txt", "TrainFiles/FrogLemmaPosOscarTokenizerTrainText500k.txt"],
        vocab_size=30522,
        min_frequency=2, ##Standard! For Bert.. Wordpiecetrainer has standard 0 but BertWordpiece overrides it to 2
        special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"],#, "[MERGE]"], #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        limit_alphabet=1000, ##Standard!
        wordpieces_prefix="", ##Standard = "##" other: ""
    )

    tokenizer.save_model("VocabularyTokenizer")

    #        Tokenizes a piece of text into its word pieces. This uses a greedy longest-match-first algorithm to perform
    #        tokenization using the given vocabulary.

    output = tokenizer.encode("hoe wordt deze zin nu een keertje opgesplitsed")
    print(output.tokens)



#---------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------
#---------------------------------- HUGGING FACE TRANSFORMERS --------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------


random_zin = "dit is een zin zeker weten gereed lopende kinderen grenzen benaderen zorlandhuggingfacelopendekinderen"


# Load the BERT tokenizer.
print('Loading BERT tokenizer...')
tokenizer = BertTokenizer(vocab_file=vocabulary_file_to_use_in_transformers, do_lower_case=True, do_basic_tokenize=True, padding_side="right",tokenize_chinese_chars=False)

### This tokenizer does also Basic Tokenization: Whitespace, punctuation, Clean text: """Performs invalid character removal and whitespace cleanup on text."""
### PADDING SIDE don't FORGET



tokenizer.save_vocabulary("VocabularyTransformer")

print(tokenizer.all_special_tokens)
print(tokenizer.tokenize(random_zin))
print(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(random_zin)))

encoded_pad = tokenizer.encode_plus(text= random_zin, padding='max_length',max_length=10, truncation=True)
print(encoded_pad["input_ids"])
print(len(encoded_pad["input_ids"]))


print(tokenizer.tokenize("een gelijk [MERGE] heid van orches [MERGE] trale noodzaak"))
encoded_pad = tokenizer.encode_plus(text= "een gelijk [MERGE] heid van orches [MERGE] trale noodzaak", padding='max_length',max_length=20, truncation=True)
print(encoded_pad["input_ids"])
print(len(encoded_pad["input_ids"]))
print(tokenizer.convert_ids_to_tokens(encoded_pad["input_ids"]))


#---------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------- REVIEW ENCODING ------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------


if include_review_encoding is True:

    print("ENCODING REVIEWS/SENTENCES...")

    print("HEHRHEHEHRHEHE")
    print(tokenizer.get_special_tokens_mask([0,1,2,3,4,5,6,7,8,9,10],already_has_special_tokens=True))
    tokenizer.add_special_tokens({'merge_token': '[MERGE]'})
    tokenizer.save_vocabulary("VocabularyTransformer")
    print(tokenizer.get_special_tokens_mask([0,1,2,3,4,5,6,7,8,9,10],already_has_special_tokens=True))
    #Also check if even with the adjusted version the function is different (get_special_tokens_mask function)

 #    FrogMorphedMergedEuroparlTotal = FrogMorphedMergedEuroparlTotal[1000000:]
 #
 #    print(f"length of europarl: {len(FrogMorphedMergedEuroparlTotal)}")
 #
 #
 #    europarl_length = len(FrogMorphedMergedEuroparlTotal)
 #
 #
 #    for_length_europarl_sentences_ids = []
 #    europarl_sentences_ids = []
 #
 #    i = 0
 #
 #    transform_start_time = time.time()
 #
 #    for sentence in FrogMorphedMergedEuroparlTotal:
 #        europarl_sentence_ids = tokenizer.encode_plus(text = sentence, padding='max_length',max_length=fixed_len,truncation=True, add_special_tokens=True)
 #        europarl_sentence_ids = europarl_sentence_ids["input_ids"]
 #        europarl_sentence_ids_without_padding = tokenizer.encode_plus(text = sentence, add_special_tokens=True)
 #        europarl_sentence_ids_without_padding = europarl_sentence_ids_without_padding["input_ids"]
 #        europarl_sentences_ids.append(europarl_sentence_ids)
 #        if i % 1000 == 0:
 #            print("sentence encoded: " + str(i) + " of: " + str(europarl_length))
 #        i += 1
 #        if i % 10000 == 0:
 #            print("")
 #            print("Sentence: ")
 #            print(sentence)
 #            print("")
 #            print("Sentence ids: ")
 #            print(europarl_sentence_ids)
 #            print("")
 #            print("Sentence ids without padding: ")
 #            print(europarl_sentence_ids_without_padding)
 #            print("")
 #            print("Sentence tokens: ")
 #            print((tokenizer.convert_ids_to_tokens(europarl_sentence_ids)))
 #            print("")
 #            print("Sentence tokens without padding: ")
 #            print((tokenizer.convert_ids_to_tokens(europarl_sentence_ids_without_padding)))
 #            print("")
 #        for_length_europarl_sentences_ids.append(europarl_sentence_ids_without_padding)
 #
 #    transform_europarl_time = format_time(time.time() - transform_start_time)
 #
 #
 # ################
 #
 #    calculate_sentence_ids_lengths_europarl_start = time.time()
 #
 #    #### FOR EUROPARL ####
 #
 #    print("Calculating europarl sentence ids lengths")
 #
 #    sentences_ids_len = [len(x) for x in for_length_europarl_sentences_ids]
 #    pd.Series(sentences_ids_len).hist()
 #    pd.Series(sentences_ids_len).describe().to_csv("InfoBertTiny/europarl_Hashtag_1000k_2000k.csv")
 #    print(pd.Series(sentences_ids_len).describe())
 #    plt.xlabel = "Sentence Length"
 #    plt.ylabel = "Amount of sentences"
 #    #plt.legend()
 #    plt.savefig("InfoBertTiny/europarl_Hashtag_1000k_2000k.png",format='png')
 #
 #    total_calculate_sentence_ids_lengths_europarl_time = format_time(time.time() - calculate_sentence_ids_lengths_europarl_start)
 #
 #    print("")
 #    print(f"transform_europarl_Hashtag_time: {transform_europarl_time}")
 #    print(f"total_calculate_sentence_ids_lengths_europarl_Hashtag_time: {total_calculate_sentence_ids_lengths_europarl_time}")
 #    print("")
 #
 #
 #    with open('OutputFeaturesBertTiny/FrogMergeHashtag_europarl_sentences_ids_1000k_2000k_128seq.pickle', 'wb') as outputfile:
 #        pickle.dump(europarl_sentences_ids, outputfile)


 #    FrogLemmaPosEuroparlTotal = FrogLemmaPosEuroparlTotal[1500000:2000000]
 #
 #    print(f"length of europarl: {len(FrogLemmaPosEuroparlTotal)}")
 #
 #
 #    europarl_length = len(FrogLemmaPosEuroparlTotal)
 #
 #
 #    for_length_europarl_sentences_ids = []
 #    europarl_sentences_ids = []
 #
 #    i = 0
 #
 #    transform_start_time = time.time()
 #
 #    for sentence in FrogLemmaPosEuroparlTotal:
 #        europarl_sentence_ids = tokenizer.encode_plus(text = sentence, padding='max_length',max_length=fixed_len,truncation=True, add_special_tokens=True)
 #        europarl_sentence_ids = europarl_sentence_ids["input_ids"]
 #        europarl_sentence_ids_without_padding = tokenizer.encode_plus(text = sentence, add_special_tokens=True)
 #        europarl_sentence_ids_without_padding = europarl_sentence_ids_without_padding["input_ids"]
 #        europarl_sentences_ids.append(europarl_sentence_ids)
 #        if i % 1000 == 0:
 #            print("sentence encoded: " + str(i) + " of: " + str(europarl_length))
 #        i += 1
 #        if i % 10000 == 0:
 #            print("")
 #            print("Sentence: ")
 #            print(sentence)
 #            print("")
 #            print("Sentence ids: ")
 #            print(europarl_sentence_ids)
 #            print("")
 #            print("Sentence ids without padding: ")
 #            print(europarl_sentence_ids_without_padding)
 #            print("")
 #            print("Sentence tokens: ")
 #            print((tokenizer.convert_ids_to_tokens(europarl_sentence_ids)))
 #            print("")
 #            print("Sentence tokens without padding: ")
 #            print((tokenizer.convert_ids_to_tokens(europarl_sentence_ids_without_padding)))
 #            print("")
 #        for_length_europarl_sentences_ids.append(europarl_sentence_ids_without_padding)
 #
 #    transform_europarl_time = format_time(time.time() - transform_start_time)
 #
 #
 # ################
 #
 #    calculate_sentence_ids_lengths_europarl_start = time.time()
 #
 #    #### FOR EUROPARL ####
 #
 #    print("Calculating europarl sentence ids lengths")
 #
 #    sentences_ids_len = [len(x) for x in for_length_europarl_sentences_ids]
 #    pd.Series(sentences_ids_len).hist()
 #    pd.Series(sentences_ids_len).describe().to_csv("InfoBertTiny/europarl_LEMMAPOS_1500k_2000k.csv")
 #    print(pd.Series(sentences_ids_len).describe())
 #    plt.xlabel = "Sentence Length"
 #    plt.ylabel = "Amount of sentences"
 #    #plt.legend()
 #    plt.savefig("InfoBertTiny/europarl_LEMMAPOS_1500k_2000k.png",format='png')
 #
 #    total_calculate_sentence_ids_lengths_europarl_time = format_time(time.time() - calculate_sentence_ids_lengths_europarl_start)
 #
 #    print("")
 #    print(f"transform_europarl_Hashtag_time: {transform_europarl_time}")
 #    print(f"total_calculate_sentence_ids_lengths_europarl_LemmaPos_time: {total_calculate_sentence_ids_lengths_europarl_time}")
 #    print("")
 #
 #
 #    with open('OutputFeaturesBertTiny/FrogLemmaPosPiece_europarl_sentences_ids_1500k_2000k_128seq.pickle', 'wb') as outputfile:
 #        pickle.dump(europarl_sentences_ids, outputfile)

    FrogLemmaPosOscarTotal = FrogLemmaPosOscarTotal

    print(f"length of europarl: {len(FrogLemmaPosOscarTotal)}")


    europarl_length = len(FrogLemmaPosOscarTotal)


    for_length_europarl_sentences_ids = []
    europarl_sentences_ids = []

    i = 0

    transform_start_time = time.time()

    for sentence in FrogLemmaPosOscarTotal:
        europarl_sentence_ids = tokenizer.encode_plus(text = sentence, padding='max_length',max_length=fixed_len,truncation=True, add_special_tokens=True)
        europarl_sentence_ids = europarl_sentence_ids["input_ids"]
        europarl_sentence_ids_without_padding = tokenizer.encode_plus(text = sentence, add_special_tokens=True)
        europarl_sentence_ids_without_padding = europarl_sentence_ids_without_padding["input_ids"]
        europarl_sentences_ids.append(europarl_sentence_ids)
        if i % 1000 == 0:
            print("sentence encoded: " + str(i) + " of: " + str(europarl_length))
        i += 1
        if i % 10000 == 0:
            print("")
            print("Sentence: ")
            print(sentence)
            print("")
            print("Sentence ids: ")
            print(europarl_sentence_ids)
            print("")
            print("Sentence ids without padding: ")
            print(europarl_sentence_ids_without_padding)
            print("")
            print("Sentence tokens: ")
            print((tokenizer.convert_ids_to_tokens(europarl_sentence_ids)))
            print("")
            print("Sentence tokens without padding: ")
            print((tokenizer.convert_ids_to_tokens(europarl_sentence_ids_without_padding)))
            print("")
        for_length_europarl_sentences_ids.append(europarl_sentence_ids_without_padding)

    transform_europarl_time = format_time(time.time() - transform_start_time)


 ################

    calculate_sentence_ids_lengths_europarl_start = time.time()

    #### FOR EUROPARL ####

    print("Calculating europarl sentence ids lengths")

    sentences_ids_len = [len(x) for x in for_length_europarl_sentences_ids]
    pd.Series(sentences_ids_len).hist()
    pd.Series(sentences_ids_len).describe().to_csv("InfoBertTiny/oscar_LEMMAPOS_1700k_2000k.csv")
    print(pd.Series(sentences_ids_len).describe())
    plt.xlabel = "Sentence Length"
    plt.ylabel = "Amount of sentences"
    #plt.legend()
    plt.savefig("InfoBertTiny/oscar_LEMMAPOS_1700k_2000k.png",format='png')

    total_calculate_sentence_ids_lengths_europarl_time = format_time(time.time() - calculate_sentence_ids_lengths_europarl_start)

    print("")
    print(f"transform_europarl_Hashtag_time: {transform_europarl_time}")
    print(f"total_calculate_sentence_ids_lengths_europarl_LemmaPos_time: {total_calculate_sentence_ids_lengths_europarl_time}")
    print("")


    with open('OutputFeaturesBertTiny/FrogLemmaPosPiece_oscar_sentences_ids_1700k_2000k_128seq.pickle', 'wb') as outputfile:
        pickle.dump(europarl_sentences_ids, outputfile)

    ### Rename features ###






 #    FrogMorphedMergedOscarTotal = FrogMorphedMergedOscarTotal[600000:]
 #
 #    print(f"length of oscar: {len(FrogMorphedMergedOscarTotal)}")
 #
 #
 #    oscar_length = len(FrogMorphedMergedOscarTotal)
 #
 #
 #    for_length_oscar_sentences_ids = []
 #    oscar_sentences_ids = []
 #
 #    i = 0
 #
 #    transform_start_time = time.time()
 #
 #    for sentence in FrogMorphedMergedOscarTotal:
 #        oscar_sentence_ids = tokenizer.encode_plus(text = sentence, padding='max_length',max_length=fixed_len,truncation=True, add_special_tokens=True)
 #        oscar_sentence_ids = oscar_sentence_ids["input_ids"]
 #        oscar_sentence_ids_without_padding = tokenizer.encode_plus(text = sentence, add_special_tokens=True)
 #        oscar_sentence_ids_without_padding = oscar_sentence_ids_without_padding["input_ids"]
 #        oscar_sentences_ids.append(oscar_sentence_ids)
 #        if i % 1000 == 0:
 #            print("sentence encoded: " + str(i) + " of: " + str(oscar_length))
 #        i += 1
 #        if i % 10000 == 0:
 #            print("")
 #            print("Sentence: ")
 #            print(sentence)
 #            print("")
 #            print("Sentence ids: ")
 #            print(oscar_sentence_ids)
 #            print("")
 #            print("Sentence ids without padding: ")
 #            print(oscar_sentence_ids_without_padding)
 #            print("")
 #            print("Sentence tokens: ")
 #            print((tokenizer.convert_ids_to_tokens(oscar_sentence_ids)))
 #            print("")
 #            print("Sentence tokens without padding: ")
 #            print((tokenizer.convert_ids_to_tokens(oscar_sentence_ids_without_padding)))
 #            print("")
 #        for_length_oscar_sentences_ids.append(oscar_sentence_ids_without_padding)
 #
 #    transform_oscar_time = format_time(time.time() - transform_start_time)
 #
 #
 # ################
 #
 #    calculate_sentence_ids_lengths_oscar_start = time.time()
 #
 #    #### FOR EUROPARL ####
 #
 #    print("Calculating oscar sentence ids lengths")
 #
 #    sentences_ids_len = [len(x) for x in for_length_oscar_sentences_ids]
 #    pd.Series(sentences_ids_len).hist()
 #    pd.Series(sentences_ids_len).describe().to_csv("InfoBertTiny/oscar_Hashtag_1500k_2000k.csv")
 #    print(pd.Series(sentences_ids_len).describe())
 #    plt.xlabel = "Sentence Length"
 #    plt.ylabel = "Amount of sentences"
 #    #plt.legend()
 #    plt.savefig("InfoBertTiny/oscar_Hashtag_1500k_2000k.png",format='png')
 #
 #    total_calculate_sentence_ids_lengths_oscar_time = format_time(time.time() - calculate_sentence_ids_lengths_oscar_start)
 #
 #    print("")
 #    print(f"transform_oscar_time: {transform_oscar_time}")
 #    print(f"total_calculate_sentence_ids_lengths_oscar_time: {total_calculate_sentence_ids_lengths_oscar_time}")
 #    print("")
 #
 #
 #    with open('OutputFeaturesBertTiny/FrogMerge_hashtag_oscar_sentences_ids_1500k_2000k_128seq.pickle', 'wb') as outputfile:
 #        pickle.dump(oscar_sentences_ids, outputfile)
 #
 #





if include_morph_post_review is True:

    with open("TrainFiles/allNederlandseZinnenFrog.txt", "r") as text_file:
        allNederlandseZinnenFrog = text_file.read()
    allNederlandseZinnenFrogList = allNederlandseZinnenFrog.split("\n\n\n")

    for nederlandseZin in allNederlandseZinnenFrogList:
        nederlandseZin = nederlandseZin.replace(" [MERGE] ", " insertMergeToken ")
        nederlandseZin = ''.join([c for c in nederlandseZin if c not in punctuation])
        nederlandseZin = nederlandseZin.replace(" insertMergeToken "," [MERGE] ")
        print("")
        print(f"original sentence: {nederlandseZin}")
        print(f"tokenized sentence: {tokenizer.tokenize(nederlandseZin)}")

elif include_morph_Hashtag_post_review is True:
    with open("TrainFiles/allNederlandseZinnenFrog.txt", "r") as text_file:
        allNederlandseZinnenFrog = text_file.read()
    allNederlandseZinnenFrogList = allNederlandseZinnenFrog.split("\n\n\n")

    for nederlandseZin in allNederlandseZinnenFrogList:
        nederlandseZin = nederlandseZin.replace(" [MERGE] ", " insertMergeToken ")
        nederlandseZin = ''.join([c for c in nederlandseZin if c not in punctuation])
        nederlandseZin = nederlandseZin.replace(" insertMergeToken "," [MERGE] ")
        nederlandseZin = nederlandseZin.replace(" [MERGE] ", " ##")

        print("")
        print(f"original sentence: {nederlandseZin}")
        print(f"tokenized sentence: {tokenizer.tokenize(nederlandseZin)}")

elif include_polyglot_morf_post_review is True:

    with open("TrainFiles/allNederlandseZinnenPolyglot.txt", "r") as text_file:
        allNederlandseZinnenPolyglot = text_file.read()
    allNederlandseZinnenPolyglotList = allNederlandseZinnenPolyglot.split("\n\n\n")

    for nederlandseZin in allNederlandseZinnenPolyglotList:
        nederlandseZin = nederlandseZin.replace(" [MERGE] ", " insertMergeToken ")
        nederlandseZin = ''.join([c for c in nederlandseZin if c not in punctuation])
        nederlandseZin = nederlandseZin.replace(" insertMergeToken "," [MERGE] ")
        print("")
        print(f"original sentence: {nederlandseZin}")
        print(f"tokenized sentence: {tokenizer.tokenize(nederlandseZin)}")

elif include_morfessor_morf_post_review is True:

    with open("TrainFiles/allNederlandseZinnenMorfessor.txt", "r") as text_file:
        allNederlandseZinnenMorfessor = text_file.read()
    allNederlandseZinnenMorfessorList = allNederlandseZinnenMorfessor.split("\n\n\n")

    for nederlandseZin in allNederlandseZinnenMorfessorList:
        nederlandseZin = nederlandseZin.replace(" [MERGE] ", " insertMergeToken ")
        nederlandseZin = ''.join([c for c in nederlandseZin if c not in punctuation])
        nederlandseZin = nederlandseZin.replace(" insertMergeToken "," [MERGE] ")
        print("")
        print(f"original sentence: {nederlandseZin}")
        print(f"tokenized sentence: {tokenizer.tokenize(nederlandseZin)}")

elif include_poslemma_post_review is True:

    with open("TrainFiles/allNederlandseZinnenLemmaPos.txt", "r") as text_file:
        allNederlandseZinnenLemmaPos = text_file.read()
    allNederlandseZinnenLemmaPosList = allNederlandseZinnenLemmaPos.split("\n\n\n")

    for nederlandseZin in allNederlandseZinnenLemmaPosList:

        nederlandseZin = nederlandseZin.replace("'","")
        nederlandseZin = nederlandseZin.replace("-", "")
        nederlandseZin = nederlandseZin.replace("…", "")
        nederlandseZin = nederlandseZin.replace("”", "")
        nederlandseZin = nederlandseZin.replace("<mete>", "<met-e>")
        nederlandseZin = nederlandseZin.replace("<mett>", "<met-t>")
        nederlandseZin = nederlandseZin.replace("<mets>", "<met-s>")
        nederlandseZin = nederlandseZin.replace("<mvn>", "<mv-n>")
        nederlandseZin = nederlandseZin.replace("<zondern>", "<zonder-n>")
        nederlandseZin = nederlandseZin.replace("<advpron>", "<adv-pron>")
        nederlandseZin = nederlandseZin.replace(" <LET> ", " ")
        nederlandseZin = nederlandseZin.replace(" <LET>", " ")

        print("")
        print(f"original sentence: {nederlandseZin}")
        print(f"tokenized sentence: {tokenizer.tokenize(nederlandseZin)}")


else:
    randomEnglishSentence = "hey this is just a tokenized sentence do we need lowerlevel characters or different, who knows let's see if splelling mistakes ge interpreted"
    nederlandseZin = "Hey heb je je aangetekende brief wel degelijk gekregen Ik hoop van wel"
    nederlandseZin2 = "In de Europese Unie spreken ongeveer 25 miljoen mensen Nederlands als eerste taal, en een bijkomende acht miljoen als tweede taal. De Franse Westhoek en de regio rondom de Duitse stad Kleef zijn van oudsher Nederlandstalige gebieden, waar Nederlandse dialecten mogelijk nog gesproken worden door de oudste generaties. Ook in de voormalige kolonie Indonesië kunnen in sommige gebieden de oudste generaties nog Nederlands spreken. Het aantal sprekers van het Nederlands in de Verenigde Staten, Canada en Australië wordt geschat op ruim een half miljoen."
    nederlandseZin3 = "Dit is een korte nederlandse zin die getokenized moet worden"
    nederlandseZin4 = "Wordt deze korte zin optimaal gesegmenteerd? zeker?? Ja hoor..."
    nederlandseZin4 = "Wordt deze korte nederlandse zin optimaal gesegmenteerd"
    nederlandseZin5 = "Het Nederlands is een West-Germaanse taal en de officiële taal van Nederland, Suriname en een van de drie officiële talen van België. Binnen het Koninkrijk der Nederlanden is het Nederlands ook een officiële taal van Aruba, Curaçao en Sint-Maarten. Het Nederlands is de derde meest gesproken Germaanse taal. In de Europese Unie spreken ongeveer 25 miljoen mensen Nederlands als eerste taal, en een bijkomende acht miljoen als tweede taal. De Franse Westhoek en de regio rondom de Duitse stad Kleef zijn van oudsher Nederlandstalige gebieden, waar Nederlandse dialecten mogelijk nog gesproken worden door de oudste generaties. Ook in de voormalige kolonie Indonesië kunnen in sommige gebieden de oudste generaties nog Nederlands spreken. Het aantal sprekers van het Nederlands in de Verenigde Staten, Canada en Australië wordt geschat op ruim een half miljoen. Het Afrikaans, een van de officiële talen van Zuid-Afrika, is een dochtertaal van het Nederlands en beide talen zijn onderling verstaanbaar."
    nederlandseZin6 = "Wanneer het Nederlands als op zichzelf staande taal precies is ontstaan, is onbekend. Het Nederlands in zijn vroegste bekende vorm is het weinig gedocumenteerde Oudnederlands (voor 1170), dat eerst overloopt in het Middelnederlands, ook wel Diets genoemd (1170-1500), en daarna in het Nieuwnederlands. De scheiding tussen de continentale en de kustvarianten van het West-Germaans liep vóór de 5e eeuw dwars door wat nu Nederland en Noordwest-Duitsland heet. De kusttaal (in de wetenschappelijk literatuur Ingveoons genoemd) verspreidde zich aan de hand van Ingveoonse klankverschuivingen in afnemende mate in zuidoostelijke richting. De Friese en Saksische dialecten hebben op het vasteland het meest onder invloed ervan gestaan. In mindere mate hebben ook het West-Vlaams en het Hollands Ingveoonse kenmerken, dialecten die aan de basis hebben gestaan van het huidige Standaardnederlands. In Engeland werd het Angelsaksisch, ook een van de Ingveoonse talen, na de Normandische invasie (1066) sterk geromaniseerd. Alleen in het Fries bleef de kusttaal op het continent bewaard. Door de opeenvolgende Hoogduitse klankverschuivingen ontwikkelde zich tussen de 4e en de 9e eeuw in het continentale West-Germaans een verwijdering tussen het zogenaamde Nederfrankisch en Nedersaksisch aan de ene zijde en het Middelduits en Opperduits aan de andere zijde. Het Nederfrankisch zou uiteindelijk de basis worden van wat nu Nederlands is, terwijl het huidige Duits zijn basis vooral in het Opperduits heeft.[2] De taalscheiding verdiepte zich niet alleen, maar schoof ook geografisch naar het noorden op. Pas in de 16e eeuw begonnen de vele regionale talen in de gebieden waar nu Nederlands wordt gesproken aan hun ontwikkeling tot één standaardtaal. Tot dan toe kende elke regio haar eigen geschreven vorm(en) en daarin weken die in het zuidoosten (Limburg) en noordoosten (van Groningen tot de Achterhoek) het meest af. Zij vertoonden invloeden van de talen van het Hanzegebied en Münsterland en zouden later nauwelijks deelnemen aan de vorming van een algemene Nederlandse standaardtaal. Het economisch en bestuurlijk zwaartepunt in Vlaanderen, Holland en Brabant, met rond de 85% van alle Nederlandstalige inwoners van de Nederlanden, weerspiegelde zich ook in de dominantie van de geschreven varianten uit die gewesten.[3] Deze schrijftalen waren academisch omdat ze vooral op de kanselarijen van vorsten, kloosters en steden en nauwelijks door de ongeletterde bevolking werden gebruikt. Rond 1500 kwam er een streven op gang om een algemene schrijftaal te ontwikkelen die in ruimere gebieden bruikbaar kon zijn door verschillende regionale elementen in zich te verenigen. Dat was ook een behoefte vanuit de centralisering van het bestuur onder het Bourgondische hertogschap dat zijn gezag vanuit Brussel over de gehele Nederlanden wilde uitbreiden, een streven waarin keizer Karel V ten slotte ook zou slagen. In de Reformatie waren het vooral de Bijbelvertalingen en religieuze traktaten waarmee een brede verspreiding werd beoogd, en welke daarom doelbewust in een algemene schrijftaal werden gesteld. Voorlopig bleef het bij pogingen waarin elke auteur zijn eigen streektaal het meeste gewicht gaf. Als benaming voor het Nederlands gelden gedurende Middeleeuwen vooral varianten van Diets/Duuts, het woord Nederlands wordt in 1482 voor het eerst aangetroffen. In de tweede helft van de 16e eeuw komt hier, als synoniem, het woord Nederduytsch bij. Het betreft hier een samenvoeging van Nederlands/Nederlanden en Diets/Duuts (nadat de uu-klank in het Nieuwnederlands naar een ui-klank omboog) en vindt haar oorsprong bij de Rederijkers. Na 1750 neemt het gebruik van Nederduytsch gestaag af, met een duidelijke versnelling na 1815. Met uitzondering van de periode tussen 1651-1700 is Nederlands vanaf 1550 de populairste benaming voor de Nederlandse taal.[4][5][6] De gesproken taal van de hogere standen ging zich pas langzamerhand naar deze nieuwe standaardtaal richten, althans in de noordelijke Nederlanden en het eerst in Holland. Hiermee vond de scheiding in ontwikkeling plaats tussen het Nederlands in Nederland waar men de standaardtaal ook ging spreken, en Vlaanderen waar de hogere standen op het Frans overgingen. De gesproken taal van de lagere standen bleef een gewestelijke of een stedelijke variant totdat de bevolking onder de leerplicht het Nederlands als schrijftaal werd geleerd en zij na enkele generaties die taal ook kon gaan spreken. Hoe langzaam dit proces moest verlopen mag blijken uit de analfabetencijfers, tevens indicaties voor schoolbezoek, die rond 1800 in de noordelijke Nederlanden nog een derde en in Vlaanderen twee derden van de volwassen bevolking omvatten. Om de geschreven Nederlandse standaardtaal tot een dagelijkse omgangstaal te maken moest, met de school als basis, uitbreiding ontstaan van de taalgebruikfuncties. Een doorslaggevende rol speelden daarin de nationaal georganiseerde massamedia en de bovenregionale communicatie ten gevolge van een sterk toenemende bevolkingsmobiliteit."

    test_twitterReview_neg = 'heeeeel sleeecht nie aan te zien zeer matig'

    print("")
    print(f"original sentence: {nederlandseZin}")
    print(f"tokenized sentence: {tokenizer.tokenize(nederlandseZin)}")
    print("")
    print(f"original sentence: {nederlandseZin2}")
    print(f"tokenized sentence: {tokenizer.tokenize(nederlandseZin2)}")
    print("")
    print(f"original sentence: {nederlandseZin3}")
    print(f"tokenized sentence: {tokenizer.tokenize(nederlandseZin3)}")
    print("")
    print(f"original sentence: {nederlandseZin4}")
    print(f"tokenized sentence: {tokenizer.tokenize(nederlandseZin4)}")
    print("")
    print(f"original sentence: {nederlandseZin5}")
    print(f"tokenized sentence: {tokenizer.tokenize(nederlandseZin5)}")
    print("")
    print(f"original sentence: {nederlandseZin6}")
    print(f"tokenized sentence: {tokenizer.tokenize(nederlandseZin6)}")






#################################### TESTING MORPHOLOGY WORTH #####################

if TEST_MORPHOLOGY == True:


    with open('TrainFiles/words.pickle', 'rb') as inputfile:
        words = pickle.load(inputfile)

    with open('TrainFiles/splits.pickle', 'rb') as inputfile:
        splits = pickle.load(inputfile)

    with open('TrainFiles/wordsPolyglotted.pickle', 'rb') as inputfile:
        wordsPolyglotted = pickle.load(inputfile)

    with open('TrainFiles/wordsMorfessored.pickle', 'rb') as inputfile:
        wordsMorfessored = pickle.load(inputfile)

    with open('TrainFiles/wordsFrogged.pickle', 'rb') as inputfile:
        wordsFrogged = pickle.load(inputfile)

    print("oakekelz")
    print(wordsPolyglotted[:10])


    total = len(words)
    total_correct = 0

    # for i in range(len(wordsPolyglotted)):
    #     word = words[i]
    #     split = splits[i]
    #     encoded_word = tokenizer.encode_plus(text=word, add_special_tokens=False)
    #     encoded_word = encoded_word["input_ids"]
    #     print("")
    #     print(word)
    #     print(encoded_word)
    #     encoded_word = [y for y in encoded_word if y != 5]
    #     print(encoded_word)
    #
    #     tokenized_word = tokenizer.convert_ids_to_tokens(encoded_word)
    #     print(tokenized_word)
    #
    #
    #     if tokenized_word == split:
    #         print("TRUE")
    #         # print(convPolyMorph[i].split(" "))
    #         # print(splits[i])
    #         total_correct += 1
    #     else:
    #         print("FALSE")
    #
    # accuracy = total_correct / total
    #
    # print(f"total: {total}")
    # print(f"total correct: {total_correct}")
    # print(f"accuracy of polyglot on test_set: {accuracy * 100} %")



    for i in range(len(wordsMorfessored)):
        word = words[i]
        split = splits[i]
        encoded_word = tokenizer.encode_plus(text=word, add_special_tokens=False)
        encoded_word = encoded_word["input_ids"]
        print(encoded_word)
        encoded_word = [y for y in encoded_word if y != 5]
        print(encoded_word)

        tokenized_word = tokenizer.convert_ids_to_tokens(encoded_word)
        print(tokenized_word)

        if tokenized_word == split:
            print("TRUE")
            # print(convPolyMorph[i].split(" "))
            # print(splits[i])
            total_correct += 1
        else:
            print("FALSE")

    accuracy = total_correct / total

    print(f"total: {total}")
    print(f"total correct: {total_correct}")
    print(f"accuracy of morfessor on test_set: {accuracy * 100} %")



    # for i in range(len(words)):
    #     word = words[i]
    #     split = splits[i]
    #     encoded_word = tokenizer.encode_plus(text=word, add_special_tokens=False)
    #     encoded_word = encoded_word["input_ids"]
    #     print(encoded_word)
    #     encoded_word = [y for y in encoded_word if y != 5]
    #     print(encoded_word)
    #
    #     tokenized_word = tokenizer.convert_ids_to_tokens(encoded_word)
    #     print(tokenized_word)
    #
    #
    #     if tokenized_word == split:
    #         print("TRUE")
    #         # print(convPolyMorph[i].split(" "))
    #         # print(splits[i])
    #         total_correct += 1
    #     else:
    #         print("FALSE")
    #
#     accuracy = total_correct / total
#
#     print(f"total: {total}")
#     print(f"total correct: {total_correct}")
#     print(f"accuracy of wordpiece on test_set: {accuracy * 100} %")



#---------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------ COMMENTS -----------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------


##special tokens in de BertTokenizer -> model? check SpecialTokensMixin

##Don't forget left padding for LSTM, look to PreTrainedTokenizerBase: padding_side: str = "right"
    # Padding side is right by default and overridden in subclasses. If specified in the kwargs, it is changed.
    #self.padding_side = kwargs.pop("padding_side", self.padding_side)

##For BertTiny, don't forget TRUNCATION right side, but with left 512 tokens... maybe do same with LSTM actually..

## check if this is ever used:  function _run_strip_accents -----> NOPE STANDARD IS DONT HANDLE ACCENTS

##Encode is same as doing:         Same as doing ``self.convert_tokens_to_ids(self.tokenize(text))``.

##convert_tokens_to_string needs small adjustment if ever use!

##adjust basic tokenizer if you wanna adjust preprocessing in this one (but probably already doing in Tokenizers or before even)

##Tokenizer chinese characters on (standard normally) I guess

## maybe check clean_text function

## Watch out in convert lemmapos for berttiny there might be **<>** leftover adjustment from other older tokenizer


#----------------------------------------------------------------------------------------------------------------------------------------------------
#Leftovers:

#tokenizer.add_tokens(['[MERGE]'],special_tokens=True)
    #deze functie: dan wordt MERGE niet meer lower-cased en niet gesplitst denk ik.. KIJK anders is naar get_special_token mask om te zien of werkt of niet
    #Moeilijk om toe te voegen zo denk ik.. wss alles aanpassen overal en self.merge_token erin doen..

#print(tokenizer.get_special_tokens_mask([1,2,3,4,5,6,7,8,9,10],already_has_special_tokens=True))
#special_tokens_dict = {'merge_token': '[MERGE]'}
#tokenizer.add_special_tokens(special_tokens_dict)

#! Check special_tokens mask for is MERGE works --> DONE, WORKS
# print(tokenizer.merge_token)
# print(tokenizer.convert_tokens_to_ids(tokenizer.merge_token))
# print(tokenizer.convert_ids_to_tokens([5]))
# print(tokenizer.get_special_tokens_mask([5,5,6,7,5,5,5,5,4,5,2,10,20,30,1,2,3],already_has_special_tokens=True))


#-------------------
#Notes for the future:

#   DONE The sentences nederlandsezin2 etc moeten eerst geconvert worden naar mergemorph form! aanpassen in text file

##################################"DATAAAAAAAAAA######

##FROGMERGE

#For last [1000000:] europarl transform features (forgot to mention time of first 1000k so calculate)
    #transform_europarl_time: 1:02:19
    #total_calculate_sentence_ids_lengths_europarl_time: 0:00:04

#For 500k Oscar:
    #transform_oscar_time: 0:58:35
    #total_calculate_sentence_ids_lengths_oscar_time: 0:00:04

#500k-1000k
    #transform_oscar_time: 1:10:09
    #total_calculate_sentence_ids_lengths_oscar_time: 0:00: 03

#1000k-15O0k
    #transform_oscar_time: 1:10: 43
    #total_calculate_sentence_ids_lengths_oscar_time: 0:00: 06

#1500k-2000k
    #transform_oscar_time: 1:04: 24
    #total_calculate_sentence_ids_lengths_oscar_time: 0:00: 04


#FROGPIECEHASHTAG 1000k
    #transform_europarl_Hashtag_time: 0:49:58
    #total_calculate_sentence_ids_lengths_europarl_Hashtag_time: 0:00:04

# 1000k-2000k
    #transform_europarl_Hashtag_time: 0:50:11
    #total_calculate_sentence_ids_lengths_europarl_Hashtag_time: 0:00:03

#FROGPIECEHASHTAG OSCAR 500k:
    #transform_oscar_time: 0:58:38
    #total_calculate_sentence_ids_lengths_oscar_time: 0:00:04

#1000-1500k:
    #transform_oscar_time: 0:51:05
    #total_calculate_sentence_ids_lengths_oscar_time: 0:00:03

#1500k-2000k:
    #transform_oscar_time: 0:57:22
    #total_calculate_sentence_ids_lengths_oscar_time: 0:00:03

#Pure wordpiece, 1000k europarl
    #transform_europarl_WordPiece_time: 1:00:28
    #total_calculate_sentence_ids_lengths_europarl_WordPiece_time: 0:00:04

# Pure wordpiece, 500k oscar
    # transform_oscar_time: 1:10:26
    # total_calculate_sentence_ids_lengths_oscar_time: 0:00:05

# LEMMA POS 500k europarl
    #transform_europarl_Hashtag_time: 1:13:53
    #total_calculate_sentence_ids_lengths_europarl_LemmaPos_time: 0:00:03

    #500-1000
    #transform_europarl_Hashtag_time: 1:11:00
    #total_calculate_sentence_ids_lengths_europarl_LemmaPos_time: 0:00:03

    #1000-1500

    #transform_europarl_Hashtag_time: 1:14: 24
    #total_calculate_sentence_ids_lengths_europarl_LemmaPos_time: 0:00: 04

    #1500-2000

    #transform_europarl_Hashtag_time: 1:22: 02
    #total_calculate_sentence_ids_lengths_europarl_LemmaPos_time: 0:00: 06

# LEMMA POS 300k OSCAR

    #transform_europarl_Hashtag_time: 1:40:35
    #total_calculate_sentence_ids_lengths_europarl_LemmaPos_time: 0:00:03

    #transform_europarl_Hashtag_time: 1:35: 43
    #total_calculate_sentence_ids_lengths_europarl_LemmaPos_time: 0:00: 02

    #transform_europarl_Hashtag_time: 1:37: 54
    #total_calculate_sentence_ids_lengths_europarl_LemmaPos_time: 0:00: 02

    #transform_europarl_Hashtag_time: 1:35: 54
    #total_calculate_sentence_ids_lengths_europarl_LemmaPos_time: 0:00: 02

    #transform_europarl_Hashtag_time: 1:41: 14
    #total_calculate_sentence_ids_lengths_europarl_LemmaPos_time: 0:00: 02

    #transform_europarl_Hashtag_time: 1:24: 36
    #total_calculate_sentence_ids_lengths_europarl_LemmaPos_time: 0:00: 02

    #transform_europarl_Hashtag_time: 2:00: 31
    #total_calculate_sentence_ids_lengths_europarl_LemmaPos_time: 0:00: 01