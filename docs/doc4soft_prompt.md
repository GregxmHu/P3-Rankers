##  soft prompt design
* For T5：[prefix] [q] [infix] [d] [suffix]
* For RoBerta: [q] [prefix] [mask] [suffix] [d]
* prefix,infix,suffix都是用tokenizer将有意义的文本输入转化后的token_id序列 

  monoT5中的 [Query: ] [q] [Document: ] [d] [Relevant: ]，在我们的模型中可以以这样的形式出现：[***tokenizer***(Query: )] [query_id] [[ [***tokenizer***(Document: )] 
 [document_id] [***tokenizer***(=Relevant: )] 
 
 
 ## parameter
 实验的bash文件参考commands/t5.sh 的写法就好。和以前的区别就是多传了三种id。
