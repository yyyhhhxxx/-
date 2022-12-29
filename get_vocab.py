# 摘自Wisley.Wang的CSDN博客：使用huggingface的Transformers预训练自己的bert模型+FineTuning
# https://blog.csdn.net/qq_26593695/article/details/115338593

from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.normalizers import Lowercase, NFKC, Sequence
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer
import os

# 1、创建一个空的字节对编码模型
tokenizer = Tokenizer(BPE())

#2、启用小写和unicode规范化，序列规范化器Sequence可以组合多个规范化器，并按顺序执行
tokenizer.normalizer = Sequence([
    NFKC(),
    Lowercase()
])
#3、标记化器需要一个预标记化器，负责将输入转换为ByteLevel表示。
tokenizer.pre_tokenizer = ByteLevel()

# 4、添加解码器，将token令牌化的输入恢复为原始的输入
tokenizer.decoder = ByteLevelDecoder()
# 5、初始化训练器，给他关于我们想要生成的词汇表的详细信息
trainer = BpeTrainer(vocab_size=50265, 
                     show_progress=True, 
                     initial_alphabet=ByteLevel.alphabet(), 
                     special_tokens=[
                        "<s>",
                        "<pad>",
                        "</s>",
                        "<unk>",
                        "<mask>",
                    ])
# 6、开始训练我们的语料
tokenizer.train(files=["ai_unsup/ai_corpus.txt"], trainer=trainer)
# 最终得到该语料的Tonkernize，查看下词汇大小
print("Trained vocab size: {}".format(tokenizer.get_vocab_size()))
# 保存训练的tokenizer
my_path = './my_token3/'
os.makedirs(my_path)
tokenizer.model.save(my_path)

