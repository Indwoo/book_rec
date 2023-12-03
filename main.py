import pandas as pd
import nltk
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelBinarizer

nltk.download('punkt')
nltk.download('wordnet')

# book 데이터와 user_norm 데이터를 읽어옵니다.
book = pd.read_csv('book_stemming_2.csv')
user_norm = pd.read_csv('cleaned_user_norm.csv')

# Title_stem 벡터 생성
title_pipeline = Pipeline([
    ('title_extractor', TupleExtractor(0)),
    ('title_vectorizer', CountVectorizer(min_df=1, ngram_range=(1,3))),
])
title_stem_vector = title_pipeline.fit_transform(book['Title']).toarray()

# Author_stem 벡터 생성
author_pipeline = Pipeline([
    ('author_extractor', TupleExtractor(0)),
    ('author_vectorizer', CountVectorizer(min_df=2, ngram_range=(1,1))),
])
author_stem_vector = author_pipeline.fit_transform(book['Author']).toarray()

# Year_stem 벡터 생성
year_pipeline = Pipeline([
    ('year_extractor', TupleExtractor(1)),
    ('year_vectorizer', LabelBinarizer(sparse_output=True)),
])
year_stem_vector = year_pipeline.fit_transform(book['Year']).toarray()

# Publisher_stem 벡터 생성
publisher_pipeline = Pipeline([
    ('publisher_extractor', TupleExtractor(2)),
    ('publisher_vectorizer', CountVectorizer(min_df=2, ngram_range=(1,1))),
])
publisher_stem_vector = publisher_pipeline.fit_transform(book['Publisher']).toarray()

# 생성된 특징 벡터를 데이터프레임으로 변환
book_features = pd.DataFrame({
    'Title_stem': title_stem_vector.tolist(),
    'Author_stem': author_stem_vector.tolist(),
    'Year_stem': year_stem_vector.tolist(),
    'Publisher_stem': publisher_stem_vector.tolist()
})

# 특징 벡터가 추가된 book 데이터를 새로운 파일로 저장
book_with_features = pd.concat([book, book_features], axis=1)
book_with_features.to_csv('book_with_features.csv', index=False)



# 어간 추출
# stemmed_titles = []
# stemmer = nltk.stem.PorterStemmer()
# for title in book_norm['Title']:
#     stemmed_title = ' '.join([stemmer.stem(word) for word in word_tokenize(title)])
#     stemmed_titles.append(stemmed_title)
# book_norm['stemmed_title'] = stemmed_titles
#
# # 표제어 추출
# lemmatized_titles = []
# lemmatizer = WordNetLemmatizer()
# for title in book_norm['stemmed_title']:
#     lemmatized_title = ' '.join([lemmatizer.lemmatize(word) for word in word_tokenize(title)])
#     lemmatized_titles.append(lemmatized_title)
# book_norm['lemmatized_title'] = lemmatized_titles
#
# book_norm.to_csv('book_stemming_2.csv', index=False)
