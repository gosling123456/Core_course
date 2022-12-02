from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np
a = '1234'
tokenizer = Tokenizer() # 创建一个Tokenizer对象，将一个词转换为正整数
tokenizer.fit_on_texts(1) #将词编号，词频越大，编号越小
word2index = tokenizer.word_index
vocab_size=len(word2index)
#print(vocab,len(vocab))
index2word = {word2index[word]:word for word in word2index}
x_word_ids = tokenizer.texts_to_sequences(a) #将句子中的每个词转换为数字
print(x_word_ids[1])
from tensorflow.keras.preprocessing.sequence import pad_sequences
x_padded_seqs = pad_sequences(x_word_ids,truncating='post',maxlen=100)#将每个句子设置为
x_padded_seqs=np.array(x_padded_seqs)
print(x_padded_seqs[1])
#print(vocab)
#print(x_padded_seqs[2])