# SmartLab
SmartLab
https://github.com/tccnchsu/SmartLab/
SmartLab

一、循序漸進了解基礎知識

1、Simple guide to confusion matrix terminology https://www.dataschool.io/simple-guide-to-confusion-matrix-terminology/ 

您學會了什麼?

A、基础概念

   a、True positive(TP):  pridicted YES , actual YES
   
   b、True negative(TN):  pridicted NO  , actual NO
   
   c、False positive(FP): pridicted YES , actual NO
   
   d、False negative(FN): pridicted NO  , actual YES
   
B、相关术语

                          N=165	    预测NO	预测YES	
                          实际NO     TN=50	 FP=10      60
                          实际YES    FN=5	         TP=100	    105
	                             55	            110	

   a、精确度（Accuracy）：分类器预测正确的概率
   
                         （TP+TN）/总预测次数
			 
                         （100+50）165
			 
   b、错误分类率（Misclassification Rate）:  分类器预测预测错误的概率
   
                         （FP+FN）/总预测次数
			 
                          （10+5）/165
			  
   c、预测YES时， 预测正确的概率 （True positive rate）：分类器预测结果为YES，实际结果为YES
   
                           TP/ 实际中YES的总数
			   
                           100/105
			   
                           也称作“灵敏度”或“召回"
			   
   d、预测YES时， 预测错误的概率 （False Positive Rate）：分类器预测结果为YES，实际结果为NO
   
                           FP/ 实际中NO的总数
			   
                           10/60
                           
   e、预测NO时，预测正确的概率（True Negative Rate）： 分类器预测结果为NO，实际结果也为NO
   
                           TN/实际中NO的总数
			   
                           50/60
                           
   f、精确度（precision）：分类器预测结果为YES，则此次预测正确的概率
   
                           TP/预测中YES的总数
			   
                           100/110
                           
   g、实际YES的概率（Prevalence）：样本发生中，实际结果为YES的概率
   
                           YES的总数/总预测次数
			   
                           105/65
                           
   f、实际NO的概率（Null Error Rate）： 样本发生中，实际结果为NO的概率
   
                           NO的总数/总预测数
			   
                           60/165
			   
       注：准确性悖论（Accuracy Paradox）：在实际预测中，并不是一味追求准确率越高越好，例如在看病时，不能因为A病的发病率为99%，
       
                                         那就在遇到这类病人时盲目的下定是A病。
       

   g、Cohen's Kappa：基于混淆矩阵的计算，用来衡量分类的精度，K值越高，代表分类的精度越好，K值越低，则误差影响更大。
   
                     Example： (总数为n)              
		                              实际类别
                                  预测类别   A	 B	C
                                     A      a	    b	    c
                                     B	    d	    e	    f
                                     C	    g	    h	    i
				     
	             由K的公式：k=(Po-Pe)/(1-Pe )
		     
		               其中Po=(a+e+i)/n
			       
			          Pe=[(a+d+g)*（a+b+c）+(b+e+h)*（d+e+f）+(g+h+i)*(c+f+i)]/(n*n)
				  
				  即可求出K
	       
   h、F Score ：用来测试准确度的度量，它考虑测试的精度以及召回率r。最佳是1，最差是0。它被定义为精确率和召回率的调和平均数
   
                   （1）  公式为：F（β） = (1+β*β)*(精确度*召回率）/[（β*β*精确度）+召回率]
                                        其中，X可取1、2、0.5。

                          当X=1时， F（1） = 2* (精确度*召回率）/（精确度+召回率）
			  
			            认为精确度与召回率的权重是相同的。
  
                          当X=2时，召回率的权重高于精确度
  
                          当X=0.5时，召回率的权重低于精确度

                   （2） 类型I和类型II错误的公式
		               
			       F（β）= （1+β*β）*TP / [（1+β*β）*TP + β*β*FN + FP]
			       
	           （3） 有效性衡量标准：
		   
		   公式：E= 1 - 1/[α/p + （1-α）/r]   其中p为精确度，r为召回率
		   
		   F（β） = 1 - E  ， 其中α = 1/（1 + β*β）

   i、 ROC曲线 （x轴为FP , y轴为TP）：ROC曲线是“可视化二进制分类器性能”中常用的方法，
   
2、Evaluation of binary classifiers https://en.wikipedia.org/wiki/Evaluation_of_binary_classifiers 
您學會了什麼?

3、Machine Learning Fundamentals: Sensitivity and Specificity https://www.youtube.com/watch?v=sunUKFXMHGk 
您學會了什麼?

4、reconstruct a 2X2 confusion matrix (TP, TN, FP, FN) from Sensitivity and Specificity https://stats.stackexchange.com/questions/370125/reconstruct-a-2x2-confusion-matrix-tp-tn-fp-fn-from-sensitivity-and-specifi 
您學會了什麼?

5、ROC and AUC, Clearly Explained https://www.youtube.com/watch?v=xugjARegisk
您學會了什麼? 您學會了畫ROC曲線嗎?


6.Machine Learning Fundamentals: Bias and Variance https://www.youtube.com/watch?v=EuBBz3bI-aA 您學會了什麼?



Data Clustering using Particle Swarm Optimization
(https://pdfs.semanticscholar.org/ad9a/9f90da491c4060ae2e770e73f40242878288.pdf)
注释：关于minboj_fun的优化：
      思路： 关于样本点分成2群，群中心分别为C1,C2.
             判断这两个中心选取是否合适，我们的思路是，通过对两个样本中心之间，距离来优化。
             
             
   背景：
       国内：
       （论文链接，中心思想）
       国外：
      
      minboj_fun = objFun(c1,c2) + 



优化网页：Basic writing and formatting syntax（https://help.github.com/en/articles/basic-writing-and-formatting-syntax）
参考：鐘建民 https://github.com/misakimeidaisuki/num_py_homework/blob/master/test5_10730612.ipynb
