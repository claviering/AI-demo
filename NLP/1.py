import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer

sentences = [
  'I love my dog',
  'I love my cat',
  'You love my dog!', # 会忽略感叹号
]
tokenizer = Tokenizer(num_words = 100) # 保留频率最多100个关键词
tokenizer.fix_on_texts(sentences)
word_index = tokenizer.word_index
print(word_index)