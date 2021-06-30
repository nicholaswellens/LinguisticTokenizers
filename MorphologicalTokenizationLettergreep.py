from HuggingfaceTokenization.OverrideClasses import BertWordPieceTokenizerWhitespacePreTokenizer
from transformersAdjusted import BertTokenizer

import matplotlib.pyplot as plt
import statistics
import collections



import pickle
import numpy as np
import pandas as pd
from string import punctuation

#------------------------------------------------------------

fixed_len = 30
split_frac = 0.9 #splitting fraction between TRAIN and TEST data
include_vocabulary_training = False
include_review_encoding = True

include_morph_post_review = False

#vocabulary_file_to_use_in_transformers = "VocabularyTokenizer/vocabFrogWhitePretokenizer30000.txt"  #"VocabularyTokenizer/vocabFrogWhitePretokenizer30000.txt" if you want to produce new one in Tokenizer and directly use it
#vocabulary_file_to_use_in_transformers = "VocabularyTokenizer/vocabPolyglotWhitePretokenizer30000.txt"
#vocabulary_file_to_use_in_transformers = "VocabularyTokenizer/vocabMorfessorWhitePretokenizer30000.txt"
#vocabulary_file_to_use_in_transformers = "VocabularyTokenizer/vocabLemmaPosWhitePretokenizer30000.txt"
vocabulary_file_to_use_in_transformers = "VocabularyTokenizer/vocabFrogHashtagWhitePretokenizer30000.txt"

#------------------------------------------------------------
#------------------------------------------------------------
#------------------------------------------------------------



with open('TrainFiles/FrogMorphedTrainLettergreep.pickle', 'rb') as inputfile:
    FrogMorphedTrainLettergreep = pickle.load(inputfile)

#FrogMorphedTestLettergreep = change_text_to_morphs(lettergreep_test_words,  save = True, filename = 'TrainFiles/FrogMorphedTestLettergreep.pickle')
with open('TrainFiles/FrogMorphedTestLettergreep.pickle', 'rb') as inputfile:
    FrogMorphedTestLettergreep = pickle.load(inputfile)

print(FrogMorphedTrainLettergreep[:15])

allFrogMorphedLettergreep = FrogMorphedTrainLettergreep + FrogMorphedTestLettergreep

print("lennnn")
print(len(allFrogMorphedLettergreep))

data = pd.read_csv("TrainFiles/lettergrepen_Official3.csv")
lettergreep_words = list(data["woorden"])
lettergreep_splits = list(data["lettergrepen"])
lettergreep_labels = list(data["aantal lettergrepen"])

len_dataset = len(lettergreep_words)







#-----------------------------Delete Examples with punctuation (both after FROG and in normal for EACH) !FOR LABELS MOSTLY! ---------

### Ree "#Name?" words..
i=0
while i <  len(lettergreep_words):
    if i < len(lettergreep_words):
        if lettergreep_words[i] == "#NAME?":
            print("ok")
            # NOTTT allFrogMorphedLettergreep.pop(i) BECAUSE ALREADY REMOVED
            lettergreep_words.pop(i)
            lettergreep_labels.pop(i)
            lettergreep_splits.pop(i)
        else:
            i+=1

i=0
while i < len(allFrogMorphedLettergreep):
    if i < len(allFrogMorphedLettergreep):
        containsFrog = [c for c in allFrogMorphedLettergreep[i] if c in punctuation]
        containsNormal = [c for c in lettergreep_words[i] if c in punctuation]
        if len(containsFrog) > 0 or len(containsNormal) > 0:
            #print(contains)
            allFrogMorphedLettergreep.pop(i)
            lettergreep_words.pop(i)
            lettergreep_labels.pop(i)
            lettergreep_splits.pop(i)
        else:
            i+=1

print(len(lettergreep_words))
print("equal?")
print(len(allFrogMorphedLettergreep))


#------------------------------------------------------------------

with open('TrainFiles/afterPunctuationMergeFrogMorphedLettergreep.pickle', 'rb') as inputfile:
    afterPunctuationMergeFrogMorphedLettergreep = pickle.load(inputfile)

afterPunctuationHashtagFrogMorphedLettergreep = afterPunctuationMergeFrogMorphedLettergreep
for i in range(len(afterPunctuationHashtagFrogMorphedLettergreep)):
    afterPunctuationHashtagFrogMorphedLettergreep[i] = afterPunctuationHashtagFrogMorphedLettergreep[i].replace(" [MERGE] ", " ##")

print(afterPunctuationHashtagFrogMorphedLettergreep[:2])

with open('TrainFiles/afterPunctuationMergePolyglotMorfedLettergreep.pickle', 'rb') as inputfile:
    afterPunctuationMergePolyglotMorfedLettergreep = pickle.load(inputfile)

with open('TrainFiles/afterPunctuationMergeMorfessorMorfedLettergreep.pickle', 'rb') as inputfile:
    afterPunctuationMergeMorfessorMorfedLettergreep = pickle.load(inputfile)

with open('TrainFiles/afterPunctuationLemmaPosLettergreep.pickle', 'rb') as inputfile:
    afterPunctuationLemmaPosLettergreep = pickle.load(inputfile)




len_dataset = len(lettergreep_words)
print(len_dataset)
print(len(lettergreep_splits))
print(len(lettergreep_labels))


afterPunctuationHashtagFrogMorphedTrainLettergreep = afterPunctuationHashtagFrogMorphedLettergreep[0:int(split_frac*len_dataset)]
lettergreep_train_labels = lettergreep_labels[0:int(split_frac*len_dataset)]

afterPunctuationHashtagFrogMorphedTestLettergreep = afterPunctuationHashtagFrogMorphedLettergreep[int(split_frac*len_dataset):]
lettergreep_test_labels = lettergreep_labels[int(split_frac*len_dataset):]

##analysis

print(f"Average number of lettergrepen train: {statistics.mean(lettergreep_train_labels)}")
print(f"Average number of lettergrepen test: {statistics.mean(lettergreep_test_labels)}")

#IPython ; IPython.embed() ; exit(1)

print("Number of train examples")
print(len(afterPunctuationHashtagFrogMorphedTrainLettergreep))
print("")
print("Number of test examples")
print(len(afterPunctuationHashtagFrogMorphedTestLettergreep))
#########################################

print("Duplicates: ")
print([item for item, count in collections.Counter(lettergreep_words).items()if count >1])

# print("How many with punctuation: ")
# i=0
# for word in afterPunctuationMergeFrogMorphedLettergreep:
#     contains = [c for c in word if c in punctuation]
#     if len(contains) >0:
#         i+=1
# print(i)
# print("with")

print("")
print("ANALYSIS ANALYSIS")
print(f"length of lettergreep_words: {len(lettergreep_words)}")
print(f"length of afterPunctuationHashtagFrogMorphedLettergreep: {len(afterPunctuationHashtagFrogMorphedLettergreep)}")
print("")
print(afterPunctuationHashtagFrogMorphedLettergreep[234])
print(lettergreep_words[234])
print(afterPunctuationHashtagFrogMorphedLettergreep[1234])
print(lettergreep_words[1234])
print(afterPunctuationHashtagFrogMorphedLettergreep[11234])
print(lettergreep_words[11234])
print(afterPunctuationHashtagFrogMorphedLettergreep[500])
print(lettergreep_words[500])
print(afterPunctuationHashtagFrogMorphedLettergreep[546:556])
print(lettergreep_words[546:556])
print("FINISHED FINISHED ANALYSIS ANALYSIS")

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

    # NOTE: pretokenization and postprocessing don't matter here because I use only the tokenizer to construct a vocabulary.

    tokenizer = BertWordPieceTokenizerWhitespacePreTokenizer( ##change whitespace and whitespace_split (lemmapos) accordingly
        lowercase=True,
        wordpieces_prefix = "",
        handle_chinese_chars = False
    )

    # And then train
    tokenizer.train(
        "TrainFiles/allFrogMorphTrainText.txt",
        vocab_size=30000,
        min_frequency=2, ##Standard! For Bert.. Wordpiecetrainer has standard 0 but BertWordpiece overrides it to 2
        special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "[MERGE]"],
        limit_alphabet=1000, ##Standard!
        wordpieces_prefix="", ##Standard = "##"
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
    #train_reviews_int = np.zeros((len(afterPunctuationMergeFrogMorphedTrainLettergreep), fixed_len), dtype=int)
    #train_reviews_int = np.zeros((len(afterPunctuationMergePolyglotMorfedTrainLettergreep), fixed_len), dtype=int)
    #train_reviews_int = np.zeros((len(afterPunctuationMergeMorfessorMorfedTrainLettergreep), fixed_len), dtype=int)
    #train_reviews_int = np.zeros((len(afterPunctuationLemmaPosTrainLettergreep), fixed_len), dtype=int)
    train_reviews_int = np.zeros((len(afterPunctuationHashtagFrogMorphedTrainLettergreep), fixed_len), dtype=int)

    i = 0

    for review in afterPunctuationHashtagFrogMorphedTrainLettergreep:
        train_review_int = tokenizer.encode_plus(text = review, padding='max_length',max_length=fixed_len,truncation=True, add_special_tokens=False)
        train_review_int = train_review_int["input_ids"]
        train_review_int_without_padding = tokenizer.encode_plus(text = review, add_special_tokens=False)
        train_review_int_without_padding = train_review_int_without_padding["input_ids"]

        if len(train_review_int_without_padding) > fixed_len:
            print("LONGER THAN LONGER THAN FIXED LENGTH")
            raise ZeroDivisionError
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
    #test_reviews_int = np.zeros((len(afterPunctuationMergeFrogMorphedTestLettergreep), fixed_len), dtype=int)
    #test_reviews_int = np.zeros((len(afterPunctuationMergePolyglotMorfedTestLettergreep), fixed_len), dtype=int)
    #test_reviews_int = np.zeros((len(afterPunctuationMergeMorfessorMorfedTestLettergreep), fixed_len), dtype=int)
    #test_reviews_int = np.zeros((len(afterPunctuationLemmaPosTestLettergreep), fixed_len), dtype=int)
    test_reviews_int = np.zeros((len(afterPunctuationHashtagFrogMorphedTestLettergreep), fixed_len), dtype=int)

    i=0

    for review in afterPunctuationHashtagFrogMorphedTestLettergreep:
        test_review_int = tokenizer.encode_plus(text = review, padding='max_length',max_length=fixed_len,truncation=True, add_special_tokens=False)
        test_review_int = test_review_int["input_ids"]
        test_review_int_without_padding = tokenizer.encode_plus(text = review, add_special_tokens=False)
        test_review_int_without_padding = test_review_int_without_padding["input_ids"]

        if len(test_review_int_without_padding) > fixed_len:
            print("LONGER THAN LONGER THAN FIXED LENGTH")
            raise ZeroDivisionError

        test_reviews_int[i,:] = np.array(test_review_int)
        if i % 100 == 0:
            print("review encoded: " + str(i))
        i += 1
        for_length_test_reviews_int.append(test_review_int_without_padding)




    ### Calculate encoding lengths ###

    # reviews_len = [len(x) for x in for_length_train_reviews_int]
    # pd.Series(reviews_len).hist()
    # pd.Series(reviews_len).describe().to_csv("InfoLettergreep/MORFESSOR_MERGE_WP_TRAIN_LETTERGREEP_30000.csv")
    # print(pd.Series(reviews_len).describe())
    # plt.xlabel = "Review Length"
    # plt.ylabel = "Amount of reviews"
    # #plt.legend()
    # plt.savefig("InfoLettergreep/MORFESSOR_MERGE_WP_TRAIN_LETTERGREEP_30000.png",format='png')
    #
    # reviews_len = [len(x) for x in for_length_test_reviews_int]
    # pd.Series(reviews_len).hist()
    # pd.Series(reviews_len).describe().to_csv("InfoLettergreep/MORFESSOR_MERGE_WP_TEST_LETTERGREEP_30000.csv")
    # print(pd.Series(reviews_len).describe())
    # plt.xlabel = "Review Length"
    # plt.ylabel = "Amount of reviews"
    # #plt.legend()
    # plt.savefig("InfoLettergreep/MORFESSOR_MERGE_WP_TEST_LETTERGREEP_30000.png",format='png')





    ### Rename features ###

    train_labels = np.array(lettergreep_train_labels)
    test_labels = np.array(lettergreep_test_labels)
    print(train_labels[:10])
    print(test_labels[:10])

    train_features = train_reviews_int
    len_train_features = len(train_features)

    test_features = test_reviews_int
    len_test_features = len(test_features)

    np.save('OutputFeatures/FROGHASHTAG_WP_LETTERGREEP_TrainFeatures_30000_30SeqLen.npy',train_features)
    np.save('OutputFeatures/FROGHASHTAG_WP_LETTERGREEP_TestFeatures_30000_30SeqLen.npy',test_features)
    np.save('OutputFeatures/FROGHASHTAG_WP_LETTERGREEP_TrainLabels_30000_30SeqLen.npy',train_labels)
    np.save('OutputFeatures/FROGHASHTAG_WP_LETTERGREEP_TestLabels_30000_30SeqLen.npy',test_labels)






if include_morph_post_review is True:

    with open("TrainFiles/allNederlandseZinnenFrog.txt", "r") as text_file:
        allNederlandseZinnenFrog = text_file.read()
    allNederlandseZinnenFrogList = allNederlandseZinnenFrog.split("\n\n\n")

    for nederlandseZin in allNederlandseZinnenFrogList:
        nederlandseZin = ''.join([c for c in nederlandseZin if c not in punctuation])
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
#   DONE Has punctuation been removed from the lettergreep data? Because otherwise our "-" won't be modelled because tokenizer is without -
    #If so! Then need to adjust PURE version also! (but can use old version for some lr testing)
    #don't forget lettergreep adjustments like 'if len bigger than fixed_len, give error'
#   DONE Still need to split allFrogMorphLettergreep and refer to right training data
