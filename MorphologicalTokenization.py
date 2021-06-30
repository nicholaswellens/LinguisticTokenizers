import pickle
import numpy as np
from string import punctuation
import pandas as pd
import matplotlib.pyplot as plt
from HuggingfaceTokenization.OverrideClasses import BertWordPieceTokenizerWhitespacePreTokenizer
from transformersAdjusted import BertTokenizer

#------------------------------------------------------------

fixed_len = 2500
include_loading_files = False
include_vocabulary_training = False
include_review_encoding = False


include_morph_post_review = False
include_polyglot_morf_post_review = False
include_morfessor_morf_post_review = False
include_poslemma_post_review = False
include_morph_Hashtag_post_review = False

TEST_MORPHOLOGY = True

#vocabulary_file_to_use_in_transformers = "VocabularyTokenizer/vocabFrogWhitePretokenizer30000.txt"  #"VocabularyTokenizer/vocabFrogWhitePretokenizer30000.txt" if you want to produce new one in Tokenizer and directly use it
#vocabulary_file_to_use_in_transformers = "VocabularyTokenizer/vocabPolyglotWhitePretokenizer30000.txt"
vocabulary_file_to_use_in_transformers = "VocabularyTokenizer/vocabMorfessorWhitePretokenizer30000.txt"
#vocabulary_file_to_use_in_transformers = "VocabularyTokenizer/vocabLemmaPosWhitePretokenizer30000.txt"

#vocabulary_file_to_use_in_transformers = "VocabularyTokenizer/vocabFrogHashtagWhitePretokenizer30000.txt"

#------------------------------------------------------------

print("LOADING FILES...")

# with open('TrainFiles/totTrainListShuffled.data', 'rb') as filehandle:
#     # read the data as binary data stream
#     totTrainList = pickle.load(filehandle)

# with open('TrainFiles/totTestListShuffled.data', 'rb') as filehandle:
#     # read the data as binary data stream
#     totTestList = pickle.load(filehandle)


if include_loading_files == True:

    with open('TrainFiles/trainLabelListShuffled.data', 'rb') as filehandle:
        # read the data as binary data stream
        trainLabelList = pickle.load(filehandle)

    with open('TrainFiles/testLabelListShuffled.data', 'rb') as filehandle:
        # read the data as binary data stream
        testLabelList = pickle.load(filehandle)

    #Non MorphEdited, just because we don't need [MERGE] in our training data for TOKENIZER!
    with open('TrainFiles/FrogMorphedTrainReviews.pickle', 'rb') as inputfile:
        FrogMorphedTrainReviews = pickle.load(inputfile)
        for i in range(len(FrogMorphedTrainReviews)):
           FrogMorphedTrainReviews[i] = ''.join([c for c in FrogMorphedTrainReviews[i] if c not in punctuation])
           #print(i)

    with open('TrainFiles/FrogMorphedTrainReviewsMergeEdited.pickle', 'rb') as inputfile:
        FrogMorphedTrainReviewsMergeEdited = pickle.load(inputfile)
        for i in range(len(FrogMorphedTrainReviewsMergeEdited)):
            FrogMorphedTrainReviewsMergeEdited[i] = ''.join([c for c in FrogMorphedTrainReviewsMergeEdited[i] if c not in punctuation]) #Frog adds - ', so need to remove again FOR SENTIMENT, not for BERTTINY
            FrogMorphedTrainReviewsMergeEdited[i] = FrogMorphedTrainReviewsMergeEdited[i].replace(" insertmergetoken "," [MERGE] ")
        #print(FrogMorphedTrainReviewsMergeEdited)

    with open('TrainFiles/FrogMorphedTestReviewsMergeEdited.pickle', 'rb') as inputfile:
       FrogMorphedTestReviewsMergeEdited = pickle.load(inputfile)
       for i in range(len(FrogMorphedTestReviewsMergeEdited)):
           FrogMorphedTestReviewsMergeEdited[i] = ''.join([c for c in FrogMorphedTestReviewsMergeEdited[i] if c not in punctuation])
           FrogMorphedTestReviewsMergeEdited[i] = FrogMorphedTestReviewsMergeEdited[i].replace(" insertmergetoken "," [MERGE] ")
           #print(i)

    with open('TrainFiles/FrogMorphedTrainReviewsMergeEdited.pickle', 'rb') as inputfile:
        FrogMorphedTrainReviewsHashtags = pickle.load(inputfile)
        for i in range(len(FrogMorphedTrainReviewsHashtags)):
            FrogMorphedTrainReviewsHashtags[i] = ''.join([c for c in FrogMorphedTrainReviewsHashtags[i] if c not in punctuation]) #Frog adds - ', so need to remove again FOR SENTIMENT, not for BERTTINY
            FrogMorphedTrainReviewsHashtags[i] = FrogMorphedTrainReviewsHashtags[i].replace(" insertmergetoken "," [MERGE] ")
            FrogMorphedTrainReviewsHashtags[i] = FrogMorphedTrainReviewsHashtags[i].replace(" [MERGE] ", " ##")
        #print(FrogMorphedTrainReviewsMergeEdited)

    with open('TrainFiles/FrogMorphedTestReviewsMergeEdited.pickle', 'rb') as inputfile:
        FrogMorphedTestReviewsHashtags = pickle.load(inputfile)
        for i in range(len(FrogMorphedTestReviewsHashtags)):
            FrogMorphedTestReviewsHashtags[i] = ''.join([c for c in FrogMorphedTestReviewsHashtags[i] if c not in punctuation])  # Frog adds - ', so need to remove again FOR SENTIMENT, not for BERTTINY
            FrogMorphedTestReviewsHashtags[i] = FrogMorphedTestReviewsHashtags[i].replace(" insertmergetoken "," [MERGE] ")
            FrogMorphedTestReviewsHashtags[i] = FrogMorphedTestReviewsHashtags[i].replace(" [MERGE] ", " ##")


    with open('TrainFiles/PolyglotMorfTrainReviewsMergeEdited.pickle', 'rb') as inputfile:
        PolyglotMorfTrainReviewsMergeEdited = pickle.load(inputfile)
        for i in range(len(PolyglotMorfTrainReviewsMergeEdited)):
            PolyglotMorfTrainReviewsMergeEdited[i] = PolyglotMorfTrainReviewsMergeEdited[i].replace(" [MERGE] "," insertmergetoken ")
            PolyglotMorfTrainReviewsMergeEdited[i] = ''.join([c for c in PolyglotMorfTrainReviewsMergeEdited[i] if c not in punctuation])
            PolyglotMorfTrainReviewsMergeEdited[i] = PolyglotMorfTrainReviewsMergeEdited[i].replace(" insertmergetoken "," [MERGE] ")

    with open('TrainFiles/PolyglotMorfTestReviewsMergeEdited.pickle', 'rb') as inputfile:
        PolyglotMorfTestReviewsMergeEdited = pickle.load(inputfile)
        for i in range(len(PolyglotMorfTestReviewsMergeEdited)):
            PolyglotMorfTestReviewsMergeEdited[i] = PolyglotMorfTestReviewsMergeEdited[i].replace(" [MERGE] "," insertmergetoken ")
            PolyglotMorfTestReviewsMergeEdited[i] = ''.join([c for c in PolyglotMorfTestReviewsMergeEdited[i] if c not in punctuation])
            PolyglotMorfTestReviewsMergeEdited[i] = PolyglotMorfTestReviewsMergeEdited[i].replace(" insertmergetoken "," [MERGE] ")

    # with open('TrainFiles/FrogMorphedTestReviewsMergeEdited.pickle', 'rb') as inputfile:
    #    FrogMorphedTestReviewsMergeEdited = pickle.load(inputfile)
    #    for i in range(len(FrogMorphedTestReviewsMergeEdited)):
    #        FrogMorphedTestReviewsMergeEdited[i] = ''.join([c for c in FrogMorphedTestReviewsMergeEdited[i] if c not in punctuation])
    #        FrogMorphedTestReviewsMergeEdited[i] = FrogMorphedTestReviewsMergeEdited[i].replace(" insertmergetoken "," [MERGE] ")
    #        #print(i)

    with open('TrainFiles/FrogLemmaPosTrainReviewsSeperate.pickle', 'rb') as inputfile:
        FrogLemmaPosTrainReviews = pickle.load(inputfile)
        for i in range(len(FrogLemmaPosTrainReviews)):
            FrogLemmaPosTrainReviews[i] = FrogLemmaPosTrainReviews[i].replace("'","")
            FrogLemmaPosTrainReviews[i] = FrogLemmaPosTrainReviews[i].replace("-", "")
            FrogLemmaPosTrainReviews[i] = FrogLemmaPosTrainReviews[i].replace("…", "")
            FrogLemmaPosTrainReviews[i] = FrogLemmaPosTrainReviews[i].replace("”", "")
            FrogLemmaPosTrainReviews[i] = FrogLemmaPosTrainReviews[i].replace("<mete>", "<met-e>")
            FrogLemmaPosTrainReviews[i] = FrogLemmaPosTrainReviews[i].replace("<mett>", "<met-t>")
            FrogLemmaPosTrainReviews[i] = FrogLemmaPosTrainReviews[i].replace("<mets>", "<met-s>")
            FrogLemmaPosTrainReviews[i] = FrogLemmaPosTrainReviews[i].replace("<mvn>", "<mv-n>")
            FrogLemmaPosTrainReviews[i] = FrogLemmaPosTrainReviews[i].replace("<zondern>", "<zonder-n>")
            FrogLemmaPosTrainReviews[i] = FrogLemmaPosTrainReviews[i].replace("<advpron>", "<adv-pron>")
            FrogLemmaPosTrainReviews[i] = FrogLemmaPosTrainReviews[i].replace(" <LET> ", " ")
            FrogLemmaPosTrainReviews[i] = FrogLemmaPosTrainReviews[i].replace(" <LET>", " ")


    with open('TrainFiles/FrogLemmaPosTestReviewsSeperate.pickle', 'rb') as inputfile:
        FrogLemmaPosTestReviews = pickle.load(inputfile)
        for i in range(len(FrogLemmaPosTestReviews)):
            FrogLemmaPosTestReviews[i] = FrogLemmaPosTestReviews[i].replace("'","")
            FrogLemmaPosTestReviews[i] = FrogLemmaPosTestReviews[i].replace("-", "")
            FrogLemmaPosTestReviews[i] = FrogLemmaPosTestReviews[i].replace("…", "")
            FrogLemmaPosTestReviews[i] = FrogLemmaPosTestReviews[i].replace("”", "")
            FrogLemmaPosTestReviews[i] = FrogLemmaPosTestReviews[i].replace("<mete>", "<met-e>")
            FrogLemmaPosTestReviews[i] = FrogLemmaPosTestReviews[i].replace("<mett>", "<met-t>")
            FrogLemmaPosTestReviews[i] = FrogLemmaPosTestReviews[i].replace("<mets>", "<met-s>")
            FrogLemmaPosTestReviews[i] = FrogLemmaPosTestReviews[i].replace("<mvn>", "<mv-n>")
            FrogLemmaPosTestReviews[i] = FrogLemmaPosTestReviews[i].replace("<zondern>", "<zonder-n>")
            FrogLemmaPosTestReviews[i] = FrogLemmaPosTestReviews[i].replace("<advpron>", "<adv-pron>")
            FrogLemmaPosTestReviews[i] = FrogLemmaPosTestReviews[i].replace(" <LET> ", " ")
            FrogLemmaPosTestReviews[i] = FrogLemmaPosTestReviews[i].replace(" <LET>", " ")



    print(PolyglotMorfTrainReviewsMergeEdited[:1])
    print(PolyglotMorfTestReviewsMergeEdited[:1])

    with open('TrainFiles/MorfessorMorfTrainReviewsMergeEdited.pickle', 'rb') as inputfile:
        MorfessorMorfTrainReviewsMergeEdited = pickle.load(inputfile)
        for i in range(len(MorfessorMorfTrainReviewsMergeEdited)):
            MorfessorMorfTrainReviewsMergeEdited[i] = MorfessorMorfTrainReviewsMergeEdited[i].replace(" [MERGE] "," insertmergetoken ")
            MorfessorMorfTrainReviewsMergeEdited[i] = ''.join([c for c in MorfessorMorfTrainReviewsMergeEdited[i] if c not in punctuation])
            MorfessorMorfTrainReviewsMergeEdited[i] = MorfessorMorfTrainReviewsMergeEdited[i].replace(" insertmergetoken "," [MERGE] ")


    with open('TrainFiles/MorfessorMorfTestReviewsMergeEdited.pickle', 'rb') as inputfile:
        MorfessorMorfTestReviewsMergeEdited = pickle.load(inputfile)
        for i in range(len(MorfessorMorfTestReviewsMergeEdited)):
            MorfessorMorfTestReviewsMergeEdited[i] = MorfessorMorfTestReviewsMergeEdited[i].replace(" [MERGE] "," insertmergetoken ")
            MorfessorMorfTestReviewsMergeEdited[i] = ''.join([c for c in MorfessorMorfTestReviewsMergeEdited[i] if c not in punctuation])
            MorfessorMorfTestReviewsMergeEdited[i] = MorfessorMorfTestReviewsMergeEdited[i].replace(" insertmergetoken "," [MERGE] ")

    print("")
    print("ALL FILES LOADED!")
    print("")
    # print(FrogMorphedTestReviewsMergeEdited[:2])
    # print(FrogMorphedTrainReviewsMergeEdited[:2])

    # print("")
    # print("No Merge now:")
    # print("")
    # print(FrogMorphedTrainReviews[:2])



###allFrogMorphTrainText= '\n'.join(FrogMorphedTrainReviews)
###print(allFrogMorphTrainText[:2000])

###with open("TrainFiles/allFrogMorphTrainText.txt", "w") as text_file:
###    text_file.write(allFrogMorphTrainText)

#FOR Frog Morph Hashtag
# allFrogMorphHashtagTrainText= '\n'.join(FrogMorphedTrainReviewsHashtags)
# print(allFrogMorphHashtagTrainText[:2000])
#
# with open("TrainFiles/allFrogMorphHashtagTrainText.txt", "w") as text_file:
#    text_file.write(allFrogMorphHashtagTrainText)


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


    # Initialize BertWordPiece tokenizer

    #   - does Bert normalization             * removing any control characters and replacing all whitespaces by the classic one.
    #                                         * handle chinese chars by putting spaces around them.
    #                                         * strip all accents.
    #   - does Bert pretokenization           * This pre-tokenizer splits tokens on spaces, and also on punctuation. Each occurence of a punctuation character will be treated separately.
    #   - does Bert postprocessing            * This post-processor takes care of adding the special tokens needed by a Bert model: CLS and SEP

    # NOTE: pretokenization and postprocessing don't matter here because I use only the tokenizer to construct a vocabulary. NO, PRETOK MATTERS HERE TOO

    tokenizer = BertWordPieceTokenizerWhitespacePreTokenizer(
        lowercase=True,
        wordpieces_prefix = "##", #other: ""
        handle_chinese_chars = False
    )

    #!!!!!!!!!!!!!!!!!!!!! watch out MERGE (for lemmapos and morphHashtag turned off)
    # And then train
    tokenizer.train(
        "TrainFiles/allFrogMorphHashtagTrainText.txt",
        vocab_size=30000,
        min_frequency=2, ##Standard! For Bert.. Wordpiecetrainer has standard 0 but BertWordpiece overrides it to 2
        special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"],####, "[MERGE]"], !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        limit_alphabet=1000, ##Standard!
        wordpieces_prefix="##", ##Standard = "##" other: ""
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
tokenizer = BertTokenizer(vocab_file=vocabulary_file_to_use_in_transformers, do_lower_case=True, do_basic_tokenize=True, padding_side="left",tokenize_chinese_chars=False)

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

    for_length_train_reviews_int = []
    #train_reviews_int = np.zeros((len(totTrainList), fixed_len), dtype = int)
    #train_reviews_int = np.zeros((len(FrogMorphedTrainReviewsMergeEdited), fixed_len), dtype=int)
    #train_reviews_int = np.zeros((len(PolyglotMorfTrainReviewsMergeEdited), fixed_len), dtype=int)
    #train_reviews_int = np.zeros((len(MorfessorMorfTrainReviewsMergeEdited), fixed_len), dtype=int)
    train_reviews_int = np.zeros((len(FrogLemmaPosTrainReviews), fixed_len), dtype=int)

    #train_reviews_int = np.zeros((len(FrogMorphedTrainReviewsHashtags), fixed_len), dtype=int)

    i = 0

    for review in FrogLemmaPosTrainReviews:
        train_review_int = tokenizer.encode_plus(text = review, padding='max_length',max_length=fixed_len,truncation=True, add_special_tokens=False)
        train_review_int = train_review_int["input_ids"]
        train_review_int_without_padding = tokenizer.encode_plus(text = review, add_special_tokens=False)
        train_review_int_without_padding = train_review_int_without_padding["input_ids"]
        train_reviews_int[i,:] = np.array(train_review_int)
        if i % 100 == 0:
            print("review encoded: " + str(i))
        i += 1
        if i % 1000 == 0:
            print("")
            print("Sentence: ")
            print(review)
            print("")
            print("Sentence ids: ")
            print(train_review_int)
            print("")
            print("Sentence ids without padding: ")
            print(train_review_int_without_padding)
            print("")
            print("Sentence tokens: ")
            print((tokenizer.convert_ids_to_tokens(train_review_int)))
            print("")
            print("Sentence tokens without padding: ")
            print((tokenizer.convert_ids_to_tokens(train_review_int_without_padding)))
            print("")
        for_length_train_reviews_int.append(train_review_int_without_padding)


    for_length_test_reviews_int = []
    #test_reviews_int = np.zeros((len(totTestList), fixed_len), dtype=int)
    #test_reviews_int = np.zeros((len(PolyglotMorfTestReviewsMergeEdited), fixed_len), dtype=int)
    #test_reviews_int = np.zeros((len(MorfessorMorfTestReviewsMergeEdited), fixed_len), dtype=int)
    test_reviews_int = np.zeros((len(FrogLemmaPosTestReviews), fixed_len), dtype=int)

    #test_reviews_int = np.zeros((len(FrogMorphedTestReviewsHashtags), fixed_len), dtype=int)

    i=0

    for review in FrogLemmaPosTestReviews:
        test_review_int = tokenizer.encode_plus(text = review, padding='max_length',max_length=fixed_len,truncation=True, add_special_tokens=False)
        test_review_int = test_review_int["input_ids"]
        test_review_int_without_padding = tokenizer.encode_plus(text = review, add_special_tokens=False)
        test_review_int_without_padding = test_review_int_without_padding["input_ids"]
        test_reviews_int[i,:] = np.array(test_review_int)
        if i % 100 == 0:
            print("review encoded: " + str(i))
        i += 1
        for_length_test_reviews_int.append(test_review_int_without_padding)




    ### Calculate encoding lengths ###

    # reviews_len = [len(x) for x in for_length_train_reviews_int]
    # pd.Series(reviews_len).hist()
    # pd.Series(reviews_len).describe().to_csv("InfoSentiment/FROGMORPH_HASHTAG_WP_TRAIN_30000.csv")
    # print(pd.Series(reviews_len).describe())
    # plt.xlabel = "Review Length"
    # plt.ylabel = "Amount of reviews"
    # #plt.legend()
    # plt.savefig("InfoSentiment/FROGMORPH_HASHTAGS_WP_TRAIN_30000.png",format='png')
    #
    # reviews_len = [len(x) for x in for_length_test_reviews_int]
    # pd.Series(reviews_len).hist()
    # pd.Series(reviews_len).describe().to_csv("InfoSentiment/FROGMORPH_HASHTAGS_WP_TEST_30000.csv")
    # print(pd.Series(reviews_len).describe())
    # plt.xlabel = "Review Length"
    # plt.ylabel = "Amount of reviews"
    # #plt.legend()
    # plt.savefig("InfoSentiment/FROGMORPH_HASHTAGS_WP_TEST_30000.png",format='png')





    ### Rename features ###

    train_labels = np.array(trainLabelList)
    test_labels  = np.array(testLabelList)
    print(train_labels[:10])
    print(test_labels[:10])

    train_features = train_reviews_int
    len_train_features = len(train_features)

    test_features = test_reviews_int
    len_test_features = len(test_features)

    np.save('OutputFeatures/LEMMAPOS_WP_TrainFeatures_30000_2500SeqLen.npy',train_features)
    np.save('OutputFeatures/LEMMAPOS_WP_TestFeatures_30000_2500SeqLen.npy',test_features)
    np.save('OutputFeatures/LEMMAPOS_WP_TrainLabels_30000_2500SeqLen.npy',train_labels)
    np.save('OutputFeatures/LEMMAPOS_WP_TestLabels_30000_2500SeqLen.npy',test_labels)






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

