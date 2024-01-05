import os
import sys
import multiprocessing
from tqdm import tqdm
from gensim import utils
from gensim.models.word2vec import Word2Vec
from gensim.corpora import WikiCorpus
from multiprocessing import freeze_support
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence


def tokenize_tr(content,token_min_len=2,token_max_len=50,lower=True):
  if lower:
    lowerMap = {ord(u'A'): u'a',ord(u'A'): u'a',ord(u'B'): u'b',ord(u'C'): u'c',ord(u'Ç'): u'ç',ord(u'D'): u'd',ord(u'E'): u'e',ord(u'F'): u'f',ord(u'G'): u'g',ord(u'Ğ'): u'ğ',ord(u'H'): u'h',ord(u'I'): u'ı',ord(u'İ'): u'i',ord(u'J'): u'j',ord(u'K'): u'k',ord(u'L'): u'l',ord(u'M'): u'm',ord(u'N'): u'n',ord(u'O'): u'o',ord(u'Ö'): u'ö',ord(u'P'): u'p',ord(u'R'): u'r',ord(u'S'): u's',ord(u'Ş'): u'ş',ord(u'T'): u't',ord(u'U'): u'u',ord(u'Ü'): u'ü',ord(u'V'): u'v',ord(u'Y'): u'y',ord(u'Z'): u'z'}
    content = content.translate(lowerMap)
  return [
  utils.to_unicode(token) for token in utils.tokenize(content, lower=False, errors='ignore')
  if token_min_len <= len(token) <= token_max_len and not token.startswith('_')
  ]


if __name__ == '__main__':
  freeze_support()
  dumpFile = "trwiki-latest-pages-articles.xml.bz2"
  corpusFile = "trwiki-latest-pages-articles.txt"
  modelFile = "myword2vec.model"
  my_turkish_model = None
  other_turkish_model = None
  if not os.path.exists(corpusFile):
    try:
      wiki = WikiCorpus(dumpFile, tokenizer_func=tokenize_tr, dictionary="tr_TR.dic", processes=4)
    except KeyboardInterrupt:
      sys.exit(0)
      
    with open(corpusFile, 'w', encoding='utf-8') as output:
      i=0
      for text in tqdm(wiki.get_texts()):
        output.write(' '.join(text) + '\n')
        i+=1
  else:
    print("Corpus already exists")
    if not os.path.exists("myword2vec.model"):
      model = Word2Vec(LineSentence(corpusFile), vector_size=400, window=5, min_count=5, workers=multiprocessing.cpu_count()-1)
      model.wv.save_word2vec_format(modelFile, binary=True)
    else:
      print("Model already exists")
      from gensim.models import KeyedVectors
      my_turkish_model = KeyedVectors.load_word2vec_format('myword2vec.model', binary=True)
      print(my_turkish_model.most_similar(positive=["abaküs","bilgisayar"],negative=[])[:3])
      print(my_turkish_model.doesnt_match(["elma","ev", "konak", "apartman"]))
      #print(word_vectors.n_similarity(['', ''], ['', ""])
      other_turkish_model = KeyedVectors.load_word2vec_format('trmodel', binary=True)
      print(other_turkish_model.most_similar(positive=["abaküs","bilgisayar"],negative=[])[:3])
      print(other_turkish_model.doesnt_match(["elma","ev", "konak", "apartman"]))
  
  #----------------------------------------------------------------------------------
  '''
  Creating 
  '''
  import gensim.downloader as api

  turkish_models_dict = {
    "turkish_model_1": my_turkish_model,
    "turkish_model_2": other_turkish_model,
  }
  
  google_news_300                 = None
  glove_twitter_200               = None
  glove_wiki_gigaword_300         = None
  fasttext_wiki_news_subwords_300 = None

  if not os.path.exists("word2vec-google-news-300.model"):
    google_news_300=api.load("word2vec-google-news-300")
    google_news_300.save_word2vec_format("word2vec-google-news-300.model", binary=True)
    print("word2vec-google-news-300.model created")
  else:
    google_news_300 = KeyedVectors.load_word2vec_format('word2vec-google-news-300.model', binary=True)
    print("word2vec-google-news-300.model loaded")

  if not os.path.exists("glove-twitter-200.model"):
    glove_twitter_200=api.load("glove-twitter-200")
    glove_twitter_200.save_word2vec_format("glove-twitter-200.model", binary=True)
    print("glove-twitter-200.model created")
  else: 
    glove_twitter_200 = KeyedVectors.load_word2vec_format('glove-twitter-200.model', binary=True)
    print("glove-twitter-200.model loaded")

  if not os.path.exists("glove-wiki-gigaword-300.model"):
    glove_wiki_gigaword_300=api.load("glove-wiki-gigaword-300")
    glove_wiki_gigaword_300.save_word2vec_format("glove-wiki-gigaword-300.model", binary=True)
  else: 
    glove_wiki_gigaword_300 = KeyedVectors.load_word2vec_format('glove-wiki-gigaword-300.model', binary=True)
    print("glove-wiki-gigaword-300.model loaded")

  if not os.path.exists("fasttext-wiki-news-subwords-300.model"):
    fasttext_wiki_news_subwords_300=api.load("fasttext-wiki-news-subwords-300")
    fasttext_wiki_news_subwords_300.save_word2vec_format("fasttext-wiki-news-subwords-300.model", binary=True)
  else:
    fasttext_wiki_news_subwords_300 = KeyedVectors.load_word2vec_format('fasttext-wiki-news-subwords-300.model', binary=True)
    print("fasttext-wiki-news-subwords-300.model loaded")


  english_models_dict = {
    "word2vec-google-news-300"        : google_news_300,
    "glove-twitter-200"               : glove_twitter_200,
    "glove-wiki-gigaword-300"         : glove_wiki_gigaword_300,
    "fasttext-wiki-news-subwords-300" : fasttext_wiki_news_subwords_300,
  }

  print("Loaded all models.")

  test_words = {
    "sport": {
      "Türkçe": "spor",
      "İngilizce": "sport",
    },
    "economy": {
      "Türkçe": "ekonomi",
      "İngilizce": "economy",
    },
    "magazine": {
      "Türkçe": "dergi",
      "İngilizce": "magazine",
    },
    "politics": {
      "Türkçe": "siyaset",
      "İngilizce": "politics",
    }
  }

  k_list = [1, 3, 10]
  
  import pandas as pd

  if not os.path.exists("results.csv"):
    print("Creating results.csv")
    df_results = pd.DataFrame(columns=["Model", "Word", "List of k"])
    
    for k in k_list:
      print("k = ", k)
      for model_name in english_models_dict:
        print(model_name)
        model = english_models_dict[model_name]
        for word in test_words:
          list_of_k=model.most_similar(positive=[test_words[word]["İngilizce"]],topn=k)
          df_results.loc[len(df_results)] = [model_name, word, list_of_k]
          print(word)
          print(list_of_k)
          print("---------------------------------------------------")

    for k in k_list:
      print("k = ", k)
      for model_name in turkish_models_dict:
        print(model_name)
        model = turkish_models_dict[model_name]
        for word in test_words:
          list_of_k=model.most_similar(positive=[test_words[word]["Türkçe"]],topn=k)
          df_results.loc[len(df_results)] = [model_name, word, list_of_k]
          print(word)
          print(list_of_k)
          print("---------------------------------------------------")
    
    df_results.to_csv("results.csv", index=False)
  else:
    print("results.csv already exists")
    pass

  df_results = pd.read_csv("results.csv")
  lex = pd.read_csv(
    "my_Lexicon.csv", 
    delimiter = ",", 
    index_col = 0, 
    header = 0,
  )
  #index,English Word,Valence,Arousal,Dominance,Turkish Word,Joy,Anger,Sadness,Fear,Disgust

  lex = lex[["English Word","Turkish Word","Valence","Arousal","Dominance", "Joy","Anger","Sadness","Fear","Disgust"]]

  import ast
  df_final = pd.DataFrame(columns=["Model","Word","English Word","Turkish Word","Valence","Arousal","Dominance", "Joy","Anger","Sadness","Fear","Disgust"])
  counter=0
  for index, row in df_results.iterrows():
    list_of_k=ast.literal_eval(row["List of k"])
    print(row["Model"])
    for i in list_of_k:
      print(i)
      if row["Model"] == "turkish_model_1" or row["Model"] == "turkish_model_2":
        index_list=lex.index[lex["Turkish Word"]==i[0]].tolist()
        for index in index_list:
          print(k_list[len(i)])
          df_final.loc[counter, "Model"] = row["Model"]
          df_final.loc[counter, "Word"] = row["Word"]
          df_final.loc[counter, "English Word"] = lex.loc[index, "English Word"]
          df_final.loc[counter, "Turkish Word"] = lex.loc[index, "Turkish Word"]
          df_final.loc[counter, "Valence"] = lex.loc[index, "Valence"]
          df_final.loc[counter, "Arousal"] = lex.loc[index, "Arousal"]
          df_final.loc[counter, "Dominance"] = lex.loc[index, "Dominance"]
          df_final.loc[counter, "Joy"] = lex.loc[index, "Joy"]
          df_final.loc[counter, "Anger"] = lex.loc[index, "Anger"]
          df_final.loc[counter, "Sadness"] = lex.loc[index, "Sadness"]
          df_final.loc[counter, "Fear"] = lex.loc[index, "Fear"]
          df_final.loc[counter, "Disgust"] = lex.loc[index, "Disgust"]
          #df_final.loc[counter, "K Value"] = k_list[len(i)] TODO: add K value
          counter+=1
      else:
        index_list=lex.index[lex["English Word"]==i[0]].tolist()
        for index in index_list:
          print(k_list[len(i)])
          df_final.loc[counter, "Model"] = row["Model"]
          df_final.loc[counter, "Word"] = row["Word"]
          df_final.loc[counter, "English Word"] = lex.loc[index, "English Word"]
          df_final.loc[counter, "Turkish Word"] = lex.loc[index, "Turkish Word"]
          df_final.loc[counter, "Valence"] = lex.loc[index, "Valence"]
          df_final.loc[counter, "Arousal"] = lex.loc[index, "Arousal"]
          df_final.loc[counter, "Dominance"] = lex.loc[index, "Dominance"]
          df_final.loc[counter, "Joy"] = lex.loc[index, "Joy"]
          df_final.loc[counter, "Anger"] = lex.loc[index, "Anger"]
          df_final.loc[counter, "Sadness"] = lex.loc[index, "Sadness"]
          df_final.loc[counter, "Fear"] = lex.loc[index, "Fear"]
          df_final.loc[counter, "Disgust"] = lex.loc[index, "Disgust"]
          #df_final.loc[counter, "K Value"] = k_list[len(i)] # TODO: add K value
          counter+=1
    counter+=1
  df_final.to_csv("final.csv", index=False)
  