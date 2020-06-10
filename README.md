<p style="color:blue;text-align:center">内容和细节还会再补充；文中字体加蓝和加红只是为了区别，无特殊意义。</p>
> 文档主要是根据这学期学习的机器学习相关的知识进行整理；
>
> - 主要参考资料：
>
>   - 主体内容：https://github.com/shuhuai007/Machine-Learning-Session, https://space.bilibili.com/97068901/
>   - 李航, 《统计学习方法（第2版）》
>   - Christopher M. Bishop, Pattern Recognition And Machine Learning
>   - 悉尼科技大学，徐亦达：https://space.bilibili.com/327617676/
>   - 博客，https://www.cnblogs.com/pinard/
>   - 维基百科，https://zh.wikipedia.org/



<img src="F:\学习\编程语言相关\Python\img\Sklearn 机器学习框架.jpg" alt="Sklearn 机器学习框架" style="zoom: 33%;" />

<center>图0.1 解决机器学习问题流程</center>

# 一、监督学习

- 监督学习是从标注数据中学习预测模型的机器学习问题，<font color=red>本质是学习输入到输出的映射的统计规律;</font>
- 监督学习的应用主要是：<font color=blue>回归问题、分类问题和标注问题；</font>
- <font color=red>统计学习方法的三要素：模型、策略和算法。</font>

## 1.1 回归(Regression)问题 ##

- <font color=blue>回归问题用于预测输入变量<font color=red>（自变量）</font>和输出变量<font color=red>（因变量）</font>之间的关系，</font>特别是输入变量的值发生变化时输出变量值的变化；<font color=red>回归模型表示的是输入变量到输出变量之间映射的函数。</font>

<p style="color:blue;background-color:gray;">线性回归</p>
- <font color=red>通过使用最佳的拟合直线（回归线），建立$Y$与$X$之间的关系。</font>
- 表达式：<font color=red>$Y=f\text{(}w\text{)}=w\cdot x$</font>, $
    w=\left( w_1,w_2,...,w_p \right) ^T,x=\left( x_1,x_2,...,x_N \right) ^T
    $;
  - 问题：==怎么确定$w$的值==；
  
1. **最小二乘法：**(Least Square Method)
$$
L\left( \omega \right) =\sum_{i=1}^m{\lVert \omega \cdot x_i-y_i \rVert ^2},\ 其中\ \ \widehat{W}=\left( X^TX \right) ^{-1}X^TY;
$$
- 转化成优化问题：$
  \hat{w}=\underset{w}{arg\min}\ L\left( w \right) 
  $；
- 损失函数中的<font color=red>$
  w\cdot x_i
  $表示的是每个数据点预测值与观测值得垂直距离；</font>
  - 可以用R-Square来评估模型的性能；
  - 说明：
    - 自变量与因变量之间必须满足线性关系；
    - 多元线性回归存在<font color=red>多重共线性，自相关和异方差性；</font>
    - 线性回归对异常值敏感。

  - ==存在的问题==：<font color=red>当$N\gg p$不成立时，$\left( X^TX \right) ^{-1}$不能求出来。</font><font color=blue>可以用下面的三种方式来进行处理：</font>
    - 加data;
    - 进行降维处理；
    - <font color=red>进行正则化处理。</font>
      - 正则化处理，加入正则化之后的优化问题：
        - Lasso: $p\left( w \right) =\lVert w \rVert _1$, 用参数的一范数，会**产生稀疏权值矩阵**;
        - Ridge: $p\left( w \right) =\lVert w \rVert _{2}^{2}$, 用参数的二范数，防止过拟合；
        - Elastic Net: $p\left( w \right) =\lambda _1\lVert w \rVert _1+\lambda _2\lVert w \rVert _{2}^{2}$, 将一范数和二范数进行结合。

<p style="color:blue;background-color:gray;">对最小二乘法进行正则化处理，可以得到下面的回归模型。</p>
2. **套索回归 (Lasso Regression)**

- Lasso: <font color=red>L</font>east <font color=red>A</font>bsolute <font color=red>S</font>hrinkage and <font color=red>S</font>election <font color=red>O</font>perator；

- 惩罚<font color=blue>回归系数的绝对值</font>，能减少变异性和提高回归模型的准确性；

  - 优化问题变为：
    $$
    \begin{aligned}
    \hat{w}&=\underset{w}{arg\min}\ \left[ L\left( w \right) +\lambda P\left( w \right) \right]
    \\
    &=\underset{w}{arg\min}\ \left[ \sum_{i=1}^m{\lVert w\cdot x_i-y_i \rVert ^2}+\lambda \lVert w \rVert _1 \right] 
    \end{aligned}
    $$

  - <font color=red>说明：</font>
    - Lasso回归<font color=blue>将系数收缩到0，有助于特征选择</font>；
    - 是一个$L1$的正则化方法；
    - 如果一组自变量高度相关，Lasso Regression 只会选择其中的一个，其余收缩为0。

3. **岭回归 (Ridge Regression)**

   - <font color=red>在数据有多重共线性的时候使用；</font>

   - 优化问题变为：
     $$
     \begin{aligned}
     \hat{w}&=\underset{w}{arg\min}\ \left[ L\left( w \right) +\lambda P\left( w \right) \right] 
     \\
     &=\underset{w}{arg\min}\ \left[ \sum_{i=1}^m{\lVert w\cdot x_i-y_i \rVert ^2}+\lambda \lVert w \rVert _{2}^{2} \right] 
     \end{aligned}
     $$

   - <font color=red>说明：</font>
     - 岭回归缩小了系数的值，但没达到0，<font color=blue>没有特征选择；</font>
     - 使用了$L2$正则化。

<p style="color:blue;background-color:gray;">对于正则化的说明。</p>
4. **正则化**
   
   - <font color=blue>正则化方法通过限制模型的复杂度，使得复杂的模型能够在有限大小的数据集上进行训练，而不会产生严重的过拟合现象；</font>
   
   - 前面介绍了利用正则化的思想控制过拟合，这种正则化方法在一些文献中被称为<font color=red>权值衰减(weight decay)</font>；<font color=blue>这是因为它倾向于让权值向0的方向衰减，</font>除非有数据支持；在统计学中，叫做<font color=red>参数收缩(parameter shrinkage)</font>，这种方法把参数的值向0的方向收缩；
   
   - 使用一个更加<font color=red>一般化的正则化项</font>，这时正则化的误差函数的形式为：
     $$
     \frac{1}{2}\sum_{i=1}^m{\lVert w\cdot x_i-y_i \rVert ^2}+\frac{\lambda}{2}\sum_{j=1}^n{\lVert w_j \rVert ^q}
     $$
   
     <img src=".\img\不同正则化系数.png" alt="不同正则化系数" style="zoom: 33%;" />
   
     <center>图1.1 对于不同的参数q，正则化项的轮廓线 </center>

5. **$L1$正则化使得参数有稀疏解**

   - $q=1$时为$L1$正则化，统计学中称为Lasso；
     - 它的性质为：<font color=blue>如果$\lambda$充分大，那么有些系数$w_j$会变为0，从而产生了一个稀疏(sparse)模型；</font>
   - $q=2$时为$L2$正则化。

   1. **角度1：**解空间形状

      <img src="F:\学习\编程语言相关\Python\img\不同的正则化项.png" alt="不同的正则化项" style="zoom: 33%;" />

      <center>图1.2 L2正则化约束与L1正则化约束</center>
      - 从上图可知，$L2$正则项约束后的解空间是原型，$L1$正则项约束后的解空间是多边形，很显然，<font color=red>多边形与等高线首次相交的的地方为最优解，</font>在二维情况下，$L1$正则项为一个菱形，此时的最优解为顶点$(0,w)$；
            - <font color=blue>在高维的情况下，正则项会有很多突出的角，因为顶点为最优解，所以有很多参数的值为0，所以就会有稀疏解；</font>
            - <font color=red>事实上，<font color=blue>“带正则项”</font>和<font color=blue>“带约束项”</font>是等价的，</font>所以正则化将相当于给误差函数添加了一个约束。
      
   2. **角度2：**函数叠加
   
      <img src=".\img\函数叠加解释正则化.png" alt="函数叠加解释正则化" style="zoom: 67%;" />
   
      <center>图1.3 用函数叠加解释正则化</center>
      
      - 观察图1.3：
      
        - 考虑一维的情况，设图1.3中<font color=blue>棕线为原始目标函数$L(w)$的曲线图</font>，最小值点在蓝色点处，其对应的色$w^*$值非0；
      
        - 然后考虑<font color=blue>加上$L2$正则化项，目标函数变成$L(w)+Cw^2$，其函数曲线为黄色，</font>此时，最小值在黄点处，对应的$w^*$的绝对值变小了，但仍未非0；
      
        - 考虑<font color=blue>加上$L1$正则化项，目标函数变成$L(w)+C|w|$，其函数曲线为绿色</font>，此时，最小值在红点处，对应的$w$值为0，产生了稀疏性。
      
      - 对上述现象的解释：
      
        - 加入$L1$正则项后，对函数求导，正则项部分产生的倒数在原点左边部分是$-C$，在原点右边部分是$C$；
      
        - 那么<font color=blue>只要原目标函数的导数的绝对值小于$C$，那么带正则项的目标函数在原点左边部分就是递减，在原点右边部分始终是递增的，所以最小值就是在原点；</font>
      
        - 加入$L2$正则项后，在原点处的导数是0，只要原目标函数在原点的导数不为0，那么最小值就不会在原点，<font color=red>所以$L2$只有减小$w$绝对值得作用，对解空间的稀疏性没有贡献。</font>
      
   3. **角度3：**贝叶斯先验
   
      - 从贝叶斯角度的简单解释是，<font color=blue>$L1$正则化相当于对模型参数$w$引入了Laplace先验，</font><font color=red>$L2$正则化相当于引入高斯先验，</font>而Laplace先验使参数为0的可能性更大；
   
      - $L1$加入Laplace先验
   
        - Laplace概率密度分布函数：
          $$
          f\left( x|\mu ,b \right) =\frac{1}{2b}\exp \left( -\frac{\left| x-\mu \right|}{b} \right)
          $$
          <img src=".\img\Laplace概率分布图.png" alt="Laplace概率分布图" style="zoom:67%;" />
        
            <center>图1.4 Laplace分布曲线图</center>
        
        -  <font color=blue>Laplace分布集中在$\mu$附近，且b越小，数据分布就越集中。</font>
        
        - Laplace先验函数：
          $$
          P\left( \theta _i \right) =\frac{\lambda}{2}\exp \left( -\lambda |\theta _i| \right)
          $$
          
        
          -  其中$\lambda$是控制参数$\theta$集中情况的超参数，<font color=blue>$\lambda$越大参数的分布就越集中在0附近</font>；
        
        - 因为要将先验分布引入估计函数中，所以这里用极大后验概率估计：
          $$
          \begin{aligned}
          \hat{w}&=\underset{w}{arg\max}\ \left( \prod_i{P\left( Y_i|X_i,w \right)}\prod_i{P\left( w_i \right)} \right) 
          \\
          &=\underset{w}{arg\max}\ \left( \log \prod_i{P\left( Y_i|X_i,w \right)}+\log \prod_i{P\left( w_i \right)} \right) 
          \\
          &=\underset{w}{arg\max}\ \left( \sum_i{\log P\left( Y_i|X_i,w \right)}+\sum_i{\log P\left( w_i \right)} \right) 
          \end{aligned}
          $$
          
        
          - 将Laplace分布函数代入上式：
            $$
            \begin{aligned}
            \sum_i{\log P\left( w_i \right)}&=\sum_i{\log \left( \frac{\lambda}{2}\exp \left( -\lambda |w _i| \right) \right)}
            \\
            &=\sum_i{\log \frac{\lambda}{2}}-\lambda \sum_i{|w _i|}
            \\
            &=C-\lambda \sum_i{|w_i|}
            \end{aligned}
            $$
            
        - 可以得到最后的式子：
            $$
            \hat{w}=\underset{w}{arg\min}\ \left( \sum_i{\lVert f\left( X_i \right) -Y_i \rVert ^2}+\lambda \sum_i{\left| w_i \right|} \right)
            $$
          
        - <font color=red>从上面的式子可以看出，$L1$正则化就是在原问题的基础上加上Laplace先验分布。</font>
        
      - $L2$加入高斯先验
   
        - 高斯分布函数：
          $$
          p\left( w \right) =\frac{1}{\sqrt{2\pi}\sigma}\exp \left( -\frac{\left( w-\mu \right) ^2}{2\sigma ^2} \right)
          $$
          <img src=".\img\高斯分布函数.png" alt="高斯分布函数" style="zoom:80%;" />
   
           <center>图1.5 高斯分布曲线图</center>
   
        -  高斯先验函数：
          $$
          p\left( w \right) =\frac{\lambda}{\sqrt{\pi}}\exp \left( -\lambda \lVert w_i \rVert ^2 \right)
          $$
   
        -  与上面同理，应用到极大后验概率估计中，可以得到最后的优化公式：
          $$
          \hat{w}=\underset{w}{arg\min}\,\,\left( \sum_i{\lVert f\left( X_i \right) -Y_i \rVert ^2}+\lambda \sum_i{\lVert w_i \rVert ^2} \right)
          $$
          
      -   <font color=red>从上面的式子可以看出，$L2$正则化就是在原问题的基础上加上高斯先验分布。</font>
        
      - <font color=red>分析：</font>
        
        - 高斯分布在极值点（0点）处是平滑的，也就是<font color=blue>高斯先验分布认为参数$w$在极值点附近的可能性是接近的</font>，这就是$L2$正则化只会让$w$更接近0点，但是不会等于0的原因；
          - <font color=blue>Laplace先验分布在极值点（0点）处是一个尖峰，</font>所以Laplace先验分布中参数$w$取值为0的可能性要更高。

## 1.2 分类问题 ##

**线性分类**

- 硬分类，分类标签 $y\epsilon \left\{ 0,1 \right\}$：

  - <font color=blue>线性判别分析</font>(Linear/Fisher Discriminant Analysis, LDA);
    - 类间大，类内小；
    - 优化问题：$\underset{w}{\max}\ J_w=\frac{\left( \bar{z}_1-\bar{z}_2 \right) ^2}{s_1+s_2}$ , $z$为类间, $s$为类内；  

  - <font color=blue>感知机</font>(perceptron)：错误驱动，loss function：$L\left( w \right) =\sum_{x_i\in D}{-y_iw^Tx_i}$
- 软分类，分类标签 $y\in \left[ 0,1 \right]$；
  - <font color=red>生成式：学习联合概率$P\left( X,Y \right)$, 然后求出条件概率$P\left( Y|X \right)$ </font>; 
	  - 朴素贝叶斯 (Naive Bayes Classifier)：关键是朴素贝叶斯假设；
	  - 高斯判别分析(Gaussian Discriminant  Analysis)，两个先验假设：
	    - 样本数据类别$y$服从$Bernoulli\left( 0-1 \right)$分布；
	    - 不同类中的样本$x$的特征是连续的，服从高斯分布。
	
	- <font color=red>判别式：直接对$P\left( Y|X \right)$或者$f\left( x \right)$进行建模</font>；
	  - Logistic Regression
	    - 优化问题：$\underset{w}{\min}\frac{1}{N}\sum_{i=1}^N{\log \left[ 1+\exp \left( -y_iw^Tx_i \right) \right]}$；
	    - 用交叉熵做损失函数；
	    - 从广义线性回归$GLR$引入。  

## 1.3 Support vector machine ##

**判别式模型，与概率无关，$f\left( w \right) =w^Tx+b$ **

**分类算法，也做回归任务，SVR；**

<font color=red>**关键字：间隔，对偶，核技巧。**</font>

- 线性可分支持向量机(hard-margin SVM)：数据集线性可分；
  - 优化问题：$\left\{ \begin{array}{c}
    	\underset{w}{\min}\frac{1}{2}\lVert w \rVert ^2\\
      	s.t.\ y_i\left( wx_i+b \right) \ge 1,\ i=1,2,3,……,N\\
    \end{array} \right.$；
  - 可以利用<font color=red>对偶性</font>：
    - 对偶问题更容易求解
    - 自然引入核技巧；
    - $\left\{ \begin{array}{c}
      	\underset{w}{\min}\frac{1}{2}\sum_{i=1}^N{\sum_{j=1}^N{\alpha _i\alpha _jy_iy_j\left( x_i\cdot x_j \right)}-\sum_{i=1}^N{\alpha _i}}\\
        	s.t.\ \sum_{i=1}^N{\alpha _iy_i=0,\ \alpha _i\ge 0,\ i=}1,2,3,……,N\\
      \end{array} \right.$
- 线性支持向量机(soft-margin SVM)：数据集不一定是理想的线性可分的，允许小错误出现；
  - $\left\{ \begin{array}{c}
    	\underset{w}{\min}\frac{1}{2}\lVert w \rVert ^2+C\sum_{i=1}^N{\xi _i}\\
      	s.t.\,\,y_i\left( wx_i+b \right) \ge 1-\xi _i,\,\,i=1,2,3,……,N\\
    \end{array} \right.$；
  - $\left\{ \begin{array}{c}
    	\underset{w}{\min}\frac{1}{2}\sum_{i=1}^N{\sum_{j=1}^N{\alpha _i\alpha _jy_iy_j\left( x_i\cdot x_j \right)}-\sum_{i=1}^N{\alpha _i}}\\
      	s.t.\,\,\sum_{i=1}^N{\alpha _iy_i=0,\,\,0\le \alpha _i\le C,\,\,i=}1,2,3,……,N\\
    \end{array} \right.$
- 非线性支持向量机：引入kernel trick，将特征映射到高维空间。 
  - $\left\{ \begin{array}{c}
    	\underset{w}{\min}\frac{1}{2}\sum_{i=1}^N{\sum_{j=1}^N{\alpha _i\alpha _jy_iy_jK\left( x_i\cdot x_j \right)}-\sum_{i=1}^N{\alpha _i}}\\
      	s.t.\,\,\sum_{i=1}^N{\alpha _iy_i=0,\,\,0\le \alpha _i\le C,\,\,i=}1,2,3,……,N\\
    \end{array} \right.$。

## 1.4 决策树 ##

**基于特征对实例进行分类**

- <font color=blue>特征选择：信息增益or信息增益比；</font>
- 决策树的生成；
- 决策树的修剪。
  - ID3(迭代二叉树3代，Iterative Dichotomiser 3), Quinlan 1986
    - 用信息增益选择特征：$g\left( D,A \right) =H\left( D \right) -H\left( D|A \right)$;
    - 选择信息增益大的特征；
    - 缺点：偏向于选择值较多的特征。
  
  - C4.5, Quinlan 1993
    - 用信息增益比选择特征：$g_R\left( D,A \right) =\frac{g\left( D,A \right)}{H_A\left( D \right)}$;
    - 选择信息增益比大的特征；
    - 会出现过拟合。
  
  - 用**剪枝**来解决过拟合
    - $\left\{ \begin{array}{c}
      	C_{\alpha}\left( T \right) =C\left( T \right) +\alpha \left| T \right|\\
        	C_{\alpha}\left( T_A \right) \le C_{\alpha}\left( T_B \right)\\
      \end{array} \right.$，$C_{\alpha}\left( T \right)$为损失函数，$\left| T \right|$为叶节点个数，$C_{\alpha}\left( T_A \right)$为剪之前的损失函数，$C_{\alpha}\left( T_B \right)$为剪之后的损失函数；
    - 缺点：1、每次选择最佳特征来分个数据，并按照该特征的所有值来进行划分。一旦特征别切分后该特征便无作用；
    - 2、不能处理连续特征值，只能先将连续特征转换成离散型。
  
  - CART(classification and regression tree), Friedman & Breiman 1984
  
    - 过程
      - 生成：递归地构建二叉决策树的过程；
      - 剪枝。
  
    - 回归树：用平方误差最小化准则；
    - 分类树
      - 基尼指数：
        $$
    Gini\left( D,A \right) =\frac{\left| D_1 \right|}{\left| D \right|}Gini\left( D_1 \right) +\frac{\left| D_2 \right|}{\left| D \right|}Gini\left( D_1 \right) ,\ Gini\left( D \right) =1-\sum_{k=1}^K{\left( \frac{\left| C_k \right|}{\left| D \right|} \right) ^2};
        $$
      
      - 基尼指数越大，样本集合的不确定性越大。
      
    - CART剪枝：
      - 从产生的决策树$T_0$低端开始不断剪枝，直到$T_0$的根节点，生成$\left[ T_0,T_1,……,T_n \right]$;
      - 用交叉验证法，选择最优子树。

## 1.5 标注(tagging)问题

- 标注问题是分类问题的一个推广，可以看成是更复杂的结构预测问题的简单形式；
- <font color=blue>标注问题的输入是一个观测序列，输出是一个标记序列或状态序列；</font>
- 标注问题在信息抽取、自然语言处理等领域被广泛应用；
- 常用的方法有隐马尔可夫模型，条件随机场。



## 1.6 集成学习(ensemble learning)

- <font color=red>通过构建并结合多个机器学习器来完成学习任务；</font>
- <font color=blue>可以用于分类问题集成、回归问题集成、特征迭代集成和异常点选取集成。</font>

1. **思路**

   - 对于训练数据，通过训练若干个个体学习器，通过一定的结合策略，可以学习一个强学习器；

     

     <img src=".\img\集成学习.png" alt="集成学习" style="zoom:80%;" />

     <center>图1. 1 集成学习思路</center>
- <font color=red>集成学习的两个问题：</font>
       - ==如何得到个体学习器；==
       - ==选择什么策略==。
   
2. **个体学习器**
   - 所有的个体学习器是同种类的，或说是同质的(homogeneous)；
   - 所有的个体学习器不全是一个种类的，或说是异质的(heterogeneous)；
   - <font color=blue>同质个体学习器是用得最广泛的，其中用得最多的就是CART决策树和神经网络。</font>
   - <font color=blue>根据个体学习器的依赖关系可分为：</font>
     - 强依赖关系：一系列个体学习器需要串行生成，==代表：boosting==；
     - 不存在强依赖关系：个体学习器可以并行生成，==对表：bagging、Random Forest==。

3. ==**boosting**==：<font color=red>Sequential ensemble methods</font>

   1. 从训练集用初始权重训练出一个<font color=red>弱学习器1</font>;
   2. 根据弱学习器的误差率表现来更新训练样本的权重，利用其训练<font color=red>弱学习器2</font>;
   3. 重复进行，直到弱学习器数目达到设定值，再用结合策略进行整合。

   - 代表的方法：
     - Adaboost；
     - Boosting Tree: Gradient Boosting Tree；
     - Gradient Boosting: eXtreme Gradient Boosting (XGboost)。 

   <img src=".\img\boosting.png" alt="boosting" style="zoom: 50%;" />

   <center>图1.2 Boosting</center>

4. ==**bagging**==：<font color=red>Parallel ensemble methods</font>

   - 各个弱学习器之间没有依赖，可以并行生成；
   - 各个弱学习器的训练集<font color=red>通过随机采样</font>生成；
   - Bagging come from <font color=red>B</font>ootstrap <font color=red>AGG</font>regat<font color=red>ING</font>；

   1. <font color=blue>Bootstrap sampling 自主采样法</font>
      - Bootstrap 的目标是基于D生成m个新的数据集$D_i$，新数据集的大小记作$n'$，<font color=red>新数据集$D_i$是通过在原数据集采样得到，采样概率服从平均分布，</font>（有放回的采样）。

   <img src=".\img\Bagging.png" alt="Bagging" style="zoom: 50%;" />

   <center>图1.3 Bagging</center>
- Random Forest: bagging + 特征的随机选择。
  
5. **结合策略**

   假定学习到的T个弱学习器为${h_1,h_2,...,h_T}$

   1. 平均法Averaging

      - Simple Averaging: $H\left( x \right) =\frac{1}{T}\sum_{i=1}^T{h_i\left( x \right)}$
      - Weighted Averaging: $H\left( x \right) =\sum_{i=1}^T{w_ih_i\left( x \right)}$, $w_i$是$h_i(x)$的权重，$w_i\ge 0,\ \sum_{i=1}^T{w_i=1}$。

   2. 投票法Voting

      - 假设预测类别是${c_1,c_2,...,c_k}$，对于任意一个预测样本$x$, $T$个弱分类器的结果是$(h_1(x),...,h_T(x))$；

      1. 相对多数投票法, Plurality Voting
         - 少数服从多数，从$T$个预测结果中，数量最多的类别$c_i$为最终的分类类别；
      2. 绝对多数投票法, Majority Voting
         - 需要票数过半；
      3. 加权投票法, Weighted Voting
         - 每个分类器的票数要乘以一个权重。

   3. 学习法

      - 代表方法：Stacking；
      - 不是对弱学习器作简单的逻辑处理，<font color=red>而是再加上一层学习器；</font>
        - <font color=red>以弱学习器的结果作为输入，将训练集的输出作为输出，重新训练一个学习器来得到最终的结果，弱学习器称为初级学习器，用于结合的学习器称为次级学习器。</font>

      - 弱学习器和强学习器
        - 弱学习器：分类器的结果纸币随机分类好一点；
        - 强学习器：分类器的结果非常接近真值。

6. <font color=blue>**Boosting— Adaboost**</font>

   <font color=red>用于减少偏差；</font>

   - 两个问题：

     - 如何改变训练数据的权值或概率分布；

       <font color=red>$N$为训练集的大小</font>

       - <font color=blue>利用分类误差率：</font>$e_m=\sum_{i=1}^N{P\left( G_m\left( x_i \right) \ne y_i \right)}=\sum_{i=1}^N{w_{mi}I\left( G_m\left( x_i \right) \ne y_i \right)}$；

         <font color=blue>$G_m(x)$的系数：</font>$\alpha _m=\frac{1}{2}\log \frac{1-e_m}{e_m}$；

         <font color=blue>更新权值分布：</font>

         $
         D_{m+1}=\left( w_{m+1,1},...,w_{m+1,i},...,w_{m+1,N} \right) 
         $
         $
         w_{m+1,i}=\frac{w_{m,i}}{Z_m}\exp \left( -\alpha _my_iG_m\left( x_i \right) \right) ,\ Z_m=\sum_{i=1}^N{w_{mi}\exp \left( -\alpha _my_iG_m\left( x_i \right) \right)}
         $

     - 如何组合弱分类器。

       $M$为弱学习器的个数，进行加权处理：$
       f\left( x \right) =\sum_{m=1}^M{\alpha _mG_m\left( x \right)}
       $;

   - <font color=blue>注：</font>
     - 每一个基生成器的目标是为了最小化损失函数，所以<font color=red>Adaboost注重减小偏差</font>;
     - Adaboost只是对每个弱分类器作加权处理，只适合二分类任务；
       - $
         w_{m+1,i}=\left\{ \begin{array}{l}
         	\frac{w_{m,i}}{Z_m}e^{-\alpha _m},\ G_m\left( x_i \right) =y_i\ \ \text{正确分类；}\\
         	\frac{w_{m,i}}{Z_m}e^{\alpha _m},\ G_m\left( x_i \right) \ne y_i\ \ \ \text{错误分类。}\\
         \end{array} \right. 
         $
       - 误分类的样本的权值被放大：$
         e^{2\alpha _m}=\frac{1-e_m}{e_m}
         $。

7. <font color=blue>**Bagging**</font>

   <font color=red>用于减少方差；</font>

   - 两个问题：
     - 如何改变训练数据的权值或概率分布；
       - 从原始样本中抽取训练集，每轮从中使用Bootstrap（有放回）的方法抽$n$个训练样本，抽$k$个训练集，<font color=red>假设$k$个训练集相互独立。</font>
       - $k$个训练集可以得到$k$个基模型。
     - 如何组合弱分类器。
       - 对分类问题：采用Voting的方式；
       - 对回归问题：计算$k$个模型的均值。

8. <font color=blue>**Random Forest**</font>

   - <font color=red>Bagging + 决策树</font>，将多颗决策树集成，<font color=blue>由很多决策树组成，不同的决策树之间无关联。</font>

   - Three Steps:

     1. 随机抽样，训练决策树

        - ==随机有放回==地从从原始训练集中抽取$n$个训练样本，作为一个树的训练集。

     2. 假设每个样本集的特征维度为$M$，

        指定一个常数$m\le M$, ==随机地==从$M$个特征中选取$m$个特征子集，每次树进行分裂时，从这$m$个特征中<font color=red>选择最优（常用某种属性，如信息增益）</font>的；

     3.  重复2，每棵树尽最大可能生成，<font color=red>没有剪枝过程。</font>

   - ==a key problem：==<font color=red>如何选择最优的m（特征个数）</font>

     - 使用out-of-bag error (oob error);
     - Bootstrap每次约有1/3的样本不会出现在Bootstrap所采的样本集合中，没有参与决策树的建立，这1/3的数据称为袋外数据<font color=red>out-of-bag (oob)</font>，它们可以用以取代测试集误差估计方法；
     - 包含$N$个样本的原始数据集$D=\left\{ \left( x_1,y_1 \right) ,...,\left( x_N,y_N \right) \right\} $在$D$上进行$N$次有放回的抽样，会得到一个样本数目为$N$的数据集$D'$。
     - 样本在$N$次采样中始终未被采样的概率为：$\left( 1-\frac{1}{N} \right) ^N\ \Longrightarrow \ \underset{N\rightarrow \infty}{\lim}\left( 1-\frac{1}{N} \right) ^N=\frac{1}{e}\approx 36.8\%$；
       - 有36.8%的$D$中的数据未出现在$D'$中，这36.8%的数据可以用来充当测试集（称为第k棵树的oob样本）。

     - <font color=red>优点：</font>
       - <font color=red>两个随机性<font color=blue>（在step1 & step2）</font>的引入，使得Random Forest抗噪声能力强，方差小，泛化能力强，不一陷入过拟合（不需要剪枝）；</font>
       - 易于高度并行训练，训练速度快，适合处理大数据；
       - 随机选择决策树节点划分的特征，可以处理高维度的特征；
       - 对部分特征缺失不敏感；
       - 对误差率的估计是无偏的，则不需要交叉验证或设置单独的测试集来获取测试集上误差的无偏估计。
     
     - <font color=red>缺点：</font>
       - 对小数据或者低维数据可能不能产生很好的分类；
       - 随机森林给人的感觉就好像是一个黑盒子，无法控制模型内部的运行，只能在不同的参数之间选择。
   
   <img src="F:\学习\编程语言相关\Python\img\偏差和方差.png" alt="偏差和方差" style="zoom: 50%;" />
   
   <center>图1.4 偏差和方差</center>
- 偏差和方差：
     - 偏差，Bias：期望值与真实标记之间的差距，$bias^2=\left( \bar{f}\left( x \right) -y \right) ^2$；
       - <font color=blue>算法做了过多假设，使预测值出现Bias。</font>
     - 方差，Variance：使用样本训练数目相同的训练集带来的方差：$var\left( x \right) =E_D\left[ \left( f\left( x;D \right) -\bar{f}\left( x \right) \right) ^2 \right] $
       - <font color=blue>算法对训练集中小变化的敏感性产生的误差。</font>
   
- Bagging的<font color=red>每个训练集都是重采样得到的，且使用同种模型，</font>==那么每个模型都有相似的Bias和Variance，==<font color=blue>（实际上，各模型分布近似，但不独立）；</font>
     - $Bias=E\left[ \frac{\sum{X\left( i \right)}}{n} \right] =E\left[ X\left( i \right) \right] $;
     - $Variance=Variance\left( \frac{\sum{X\left( i \right)}}{n} \right) =\frac{Var\left( X\left( i \right) \right)}{n}$, <font color=red>减小了方差。</font>
       - $D(X+Y)=D(X)+D(Y)$;
       - $D(aX)=a^2D(x)$.

9. <font color=blue>**Stacking**</font>

- <font color=blue>可以看成一个集成学习方法，也可以看成一个结合策略。</font>
- 思想：
  1. 利用初级学习算法对原始数据进行学习，生成一个新的数据集$D'$;
  2. 次级学习器从$D'$中学习并得到最终输出。

- Algorithm:

  <font color=red>Input:</font>    training data  $D=\left\{ x_i,y_i \right\} _{i=1}^{m}$

  <font color=red>Output:</font> ensemble classifiers H

  <font color=red>Step1:   learn base-level classifies</font>  （可以是同质的也可以是异质的，但是一般是异质的）

  ​			 for t=1 to T do

  ​					learn $h_t$ based on D

  ​			 end for

  <font color=red>Step2:   construct new data set of predictions</font>，（应用交叉验证防止过拟合）

  ​		     for i=1 to m do

  ​					$D_h=\left\{ x_{i}^{'},y_i \right\} ,\ x_{i}^{'}=\left\{ h_1\left( x_i \right) ,...,h_T\left( x_i \right) \right\} $

  ​			 end for

  <font color=red>Step3:   learn a meta-classifier</font>

  ​			 learn H based on $D_h$

  ​			 return H

## 1.7 k近邻法

- k近邻法是一种基本分类与回归方法，1968年由Cover和Hart提出；
- <font color=red>k近邻法的输入为实例的特征向量，对应于特征空间的点；输出为实例的类别，可以取多类；</font>
  - k近邻法假设给定一个训练数据集，其中的实例类别已定，分类时，对新的实例，根据其k个最近邻的训练实例的类别，通过多数表决的方式进行预测；
  - k近邻法不具有显式的学习过程；
  - <font color=blue>k近邻法对特征空间进行划分，并作为其分类的“模型”。</font>
- k近邻法三个重要的因素：<font color=red>k值的选择、距离度量以及分类决策规则。</font>

- <font color=blue>k近邻算法：</font>

  - <font color=blue>输入：</font>训练数据集$T=\{\left( x_1,y_1 \right) ,...,\left( x_N,y_N \right) \}$, 

    其中$x_i\in \mathcal{X}\subseteq \mathbb{R}^n$为实例的特征向量，$y_i\in \mathcal{Y}=\left\{ c_1,c_2,...,c_K \right\} $为实例的类别，$i=1,2,...,N$；实例特征向量$x$；

    <font color=blue>输出：</font>实例$x$所属的类$y$。

    - 根据给定的距离度量，在训练集$T$中找出与$x$最邻近的$k$个点，涵盖这$k$个点的$x$的邻域记作$N_k(X)$；

    - 在$N_k(X)$中根据分类决策规则（如多数表决）决定$x$的类别$y$：
      $$
      y=\underset{c_j}{arg\max}\sum_{x_i\in N_k\left( x \right)}{I\left( y_i=c_j \right)},\ i=1,2,...,N;\ j=1,2,...,K
      $$
      式中$I$为指示函数，即当$y_i=c_i$时$I$为1，否则$I$为0。

    - k近邻法的特殊情况是$k=1$的情形，称为最近邻法，对于输入的实例点$x$，最近邻法将数据集中于$x$最邻近点的类作为$x$的类，<font color=red>k近邻法没有显示的学习过程。</font>

1. **k近邻模型**

   - 模型

     - <font color=blue>k近邻法对特征空间进行划分，并作为其分类的“模型”；</font>

     - 特征空间中，对每个训练实例点$x_i$，距离该点比其他点更近的所有点组成一个区域，叫做单元(cell)；每个训练实例点拥有一个单元，所有训练实例点的单元构成对特征空间的一个划分，最近邻算法将实例$x_i$的类$y_i$作为其单元中所有点的类标记(class label)。

       ![k近邻空间划分](.\img\k近邻空间划分.png)

       <center>图1.5 k近邻的特征空间划分</center>

   - 距离度量

     - 特征空间中两个实例点的距离是两个实例点相似程度的反映；
     - k近邻模型的特征空间一般是$n$维实数向量空间$\mathbb{R}^n$, 使用的距离度量一般是<font color=red>欧式距离、$L_p$距离或$Minkowski$距离。</font>

   - k值的选择

     - k值得选择会对k近邻法的结果产生重大影响；
       - 选择较小的k值，就相当于用较小的邻域中的训练实例进行预测，“学习”的近似误差会减少；<font color=blue>k值的减小就意味着整体模型变得复杂，容易发生过拟合；</font>
       - 选择较大的k值，就相当于用较大的邻域中的训练实例进行预测，优点是可以减小学习的估计误差，但缺点是学习的近似误差会增大；<font color=blue>k值的增大意味着整体的模型变得简单；</font>
       - 如果$k=N$，那么无论输入的实例是什么，都将简单地预测它属于在训练实例中最多的类，这是模型过于简单，完全忽略训练实例中的大量有用信息，<font color=red>是不可取的</font>；<font color=blue>在应用中，k值一般选取一个比较小的数值，==通常采用交叉验证法来选取最优的k值。==</font>

   - 分类决策规则

     - k近邻法中的分类决策规则往往是<font color=red>多数表决，由输入实例的k个邻近的训练实例中的==多数类决定输入实例的类==。</font>

2. **k近邻法的实现：kd树**

   - <font color=red>kd树是存储k维空间数据的树结构，这里的k和k近邻法中的k的含义不相同；</font>
   
   - k-dimension tree, 是一种二叉树，空间划分树；
   - <font color=red>实现k近邻法，主要考虑的问题是==如何对训练数据进行快速k近邻搜索==</font>；这在特征空间维数大及训练数据容量大时尤其必要；
  - k近邻法最简单的实现方法是<font color=red>线性扫描(linear scan)</font>，这时要计算输入实例与每一个训练实例的距离，但是当训练集很大时，计算非常耗时，不可行；
     - <font color=blue>为了提高k近邻搜索的效率，可以考虑使用特殊的结构存储训练数据，</font>以减少计算距离的次数，这里介绍kd tree。

   1. **构造kd树**

      - kd树是一种对k维空间中的实例点<font color=red>进行存储以便对其进行快速检索</font>的<font color=blue>树形数据结构</font>；kd树是二叉树，表示对k维空间的一个划分(partition)；
      
      - <font color=red>构造kd树相当于不断地用垂直于坐标轴的平面将k维空间切分，构成一系列的k维超矩形区域；</font>==kd树的每一个节点对应于一个k维超矩形区域==；
      
      - 构造思想：
      
        - <font color=red>通过递归的方法，不断地对k维空间进行划分，生成子节点；</font>
        
        1. 在超矩形区域（节点）上选择一个坐标轴和在此坐标轴上的一个<font color=blue>切分点</font>，确定一个超平面，这个超平面通过选定的切分点并垂直于选定的坐标轴，将当前超矩形区域切分为左右两个子区域（子节点）；
        2. 这时，实例被分到两个子区域，这个过程直到子区域内没有实例时终止（终止时的节点为叶节点）；
        3. <font color=blue>通常，依次选择坐标轴对空间切分，选择训练实例点在选定坐标轴上的中位数为切分点，这样得到的kd树是平衡的。</font>
   
   2. **搜索kd树**
   
      - 对kd树进行k近邻搜索
      
      - 给定一个目标点，搜索其最近邻：
      
        1. 首先找到包含目标点的叶结点；
        2. 然后从该叶结点出发，依次回退到父节点；
        3. 不断查找与目标点最邻近的结点，当确定不可能存在更近的结点时终止。
        
        - <font color=red>这样搜索就被限制在空间的局部区域上，效率大为提高。</font>
      
      - <font color=red>算法：</font>
      
        - 输入：已构造的kd树，目标点$x$；
        - 输出：$x$的最近邻；
        
        1. 在kd树中找出包含目标点$x$的叶节点：
           - 从根节点出发，递归地向下访问kd树；
           - 若目标点$x$当前维的坐标小于切分点的坐标，则移动到左子结点，否则移动到右子结点，直到子结点为叶节点为止；
          
        2. 以此叶结点为“当前最近点”；
        3. 递归地向上回退，在每个结点进行以下操作：
           - 如果该结点保存的实例点比当前最近点距离目标点更近，则以该实例点为“当前的最近点”；
           - 当前最近点一定存在于该结点一个子结点对应的区域；检查该子结点的父结点的另一子结点对应的区域是否有更近的点；具体地，检查另一子结点对应的区域是否以目标点位球心、以目标点与“当前的最近点”间的距离为半径的超球体相交；
           - 如果相交，可能在另一个子结点对应的区域内存在距目标点更近的点，移动到另一个子结点，接着递归地进行最近邻搜索；如果不相交，向上回退；
          
        4. 当回退到根结点时，搜索结束，最后的“当前的最近点”即为$x$的最近邻点。
        
        - <font color=red>如果实例点是随机分布的，kd树搜索的平均计算复杂度是$O(logN)$，这里$N$是实例点数目。</font>

   - <font color=red>kd树更适用于训练实例树远大于空间维度时的k近邻搜索；</font>当空间维度接近训练实例树时，它的效率会迅速下降，几乎接近线性扫描。

# 二、无监督学习 #

- 无监督学习是从无标注数据中学习数据的统计规律或者说内在结构的机器学习问题；
- <font color=red>基本问题包括聚类、降维、概率估计；</font>
- 无监督学习可以用于数据分析或者监督学习的前处理。

## 2.1 聚类 ##

- 聚类是将样本集合中相似的样本（实例）分配到相同的类，不相似的样本分配到不同的类。
- <font color=blue>利用特征的相似类或距离，将样本归到若干个类；</font>
- 最常用的两种聚类方法：
  - 层次聚类(hierarchical clustering)
    - 聚合：每个样本分到一个类，距离相近的两类合并，建立新的类；
    - 分裂：所有样本分到一个类，距离最远的样本分到两个新的类。
  - k均值(k-mean), Mac Queen 1967
    - 基于中心，通过迭代，将样本分到k个类；
    - 类别k的值事先确定；
    - 每个样本与其所属中心的距离最近。

- 聚类的核心：相似度(similarity)和距离(distance)

  - <font color=red>距离：值越小越相似</font>
- 闵可夫斯基距离(Minkowski distance)
      - $d_{ij}=\left( \sum_{k=1}^m{\left| x_{ki}-x_{kj} \right|^p} \right) ^{\frac{1}{p}},\ p\ge 1$;
      - $p=2$, 为欧式距离(Euclidean distance)；
      - $p=1$, 为曼哈顿距离(Manhattan distance)；
      - $p=\infty$, 为切比雪夫距离(Chebyshev distance)。
  
- 马哈拉诺比斯距离(Mahalanobis distance)，马氏距离
      - $d_{ij}=\left[ \left( x_i-x_j \right) ^TS^{-1}\left( x_i-x_j \right) \right] ^{\frac{1}{2}}$, $S$为样本集$X=\left( x_{ij} \right) _{m\times n}$的协方差矩阵;
      - $S$为单位矩阵，样本数据的各个分量相互独立，且各个分量的方差为0；
      - 此时马氏距离就是欧式距离，马氏距离为欧式距离的推广。
  
  - <font color=red>相似度：值越接近1，越相似</font>
    - 相关系数(correlation coefficient)
      - $r_{ij}=\frac{\sum_{k=1}^m{\left( x_{ki}-\bar{x}_i \right) \left( x_{kj}-\bar{x}_j \right)}}{\left[ \sum_{k=1}^m{\left( x_{ki}-\bar{x}_i \right) ^2\sum_{k=1}^m{\left( x_{kj}-\bar{x}_j \right) ^2}} \right] ^{\frac{1}{2}}},\,\,\bar{x}_i=\frac{1}{m}\sum_{i=1}^m{x_{ki}}$，
    - 夹角余弦
      - $s_{ij}=\frac{\sum_{k=1}^m{x_{ki}x_{kj}}}{\left[ \sum_{k=1}^m{x_{ki}^{2}}\sum_{k=1}^m{x_{kj}^{2}} \right] ^{\frac{1}{2}}}$。
  
- k-均值法，一个迭代过程

  - 硬聚类，每个样本只能属于一个类；

  - <font color=blue>n个样本划分到k个类别中，可以看成一个从样本到类别的函数；</font>

  - 策略：通过损失函数的最小化选取最优的划分，采用欧式距离；

    - 损失函数：$W\left( C \right) =\sum_{l=1}^k{\sum_{C\left( i \right) =l}{\lVert x_i-\bar{x}_l \rVert}^2}$,  $\bar{x}_l$为第$l$个类的均值或中心，$C$表示划分；

    - 可以转化成最优化问题：$C^*=\underset{C}{arg\min}W\left( C \right) =\underset{C}{arg\min}\sum_{l=1}^k{\sum_{C\left( i \right) =l}{\lVert x_i-\bar{x}_l \rVert}^2}$；

      这是一个组合优化问题：n个样本分到k个类，可能的分法：$S\left( n,k \right) =\frac{1}{k!}\sum_{l=1}^k{\left( -1 \right) ^{k-l}\left( \begin{array}{c}k\\l\\
      \end{array} \right) k^n}$；

      $S\left( n,k \right)$为指数级，NP困难问题；

    - <font color=blue>采用迭代的方法解决困难。</font>
  
  
  - 算法
    1. 确定类别个数k；
    2. 随机初始化k个类别中心，$m^{\left( 0 \right)}=\left( m_{1}^{\left( 0 \right)},\ m_{2}^{\left( 0 \right)},……,m_{k}^{\left( 0 \right)} \right)$；
    3. 计算每个样本与类中心对的距离，**原则：样本与类中心距离最小**，将样本分类；
    4. 更新每个类的中心：$m_l=\frac{\sum_{i=1}^n{I\left( C\left( i \right) =l \right) x^i}}{\sum_{i=1}^n{I\left( C\left( i \right) =l \right)}}$, $I\left( C\left( i \right) =l \right)$为指示函数，$C\left( i \right)$为类中样本的个数。

##  2.2 降维(dimensionality Reduction)  ##

- 维度灾难
  - 增加一个维度，样本数会以2的指数倍增加；
  - <font color=blue>几何角度：随着维度的增加，样本会分布集中在边缘，数据有稀疏性；在高维空间中，一个球体的大部分体积会聚集在表面附近的薄球壳上。</font>

- 降维的方法
  - 直接降维：特征选择；
  - 线性降维：PCA(Principle Component Analysis), MDS(Multidimensional Scaling)；
  - 非线性降维：流行
    - ISO map(Isometric Mapping), 等距特征映射；
    - LLE(Locally linear embedding), 局部线性嵌入。

- 主成分分析PCA(Principle Component Analysis), Pearson, 1901
  
  - **一个中心，两个基本点**
  
  - <font color=blue>利用正交变换把由线性相关变量表示的观测数据转换为由少数几个由线性无关变量表示的数据，线性无关的变量称为主成分</font>；
  - 主要用于发现数据中的基本结构，即数据中变量之间的关系，是数据分析的有力工具，也用于其他机器学习方法的前处理;
  -  基本步骤：
    1. 先对给定数据进行规范化，使得数据每一变量的平均值为0，方差为1；
    2. 对数据进行正交变换，原来由线性相关变量表示的数据，通过正交变换变成由若干个线性无关的新变量表示的数据；
  
       - 新变量是可能的<font color=blue>正交变换中的方差的和（信息保存）最大的</font>；
       - 新变量一次成为第一主成分、第二主成分等等。
  
  - PCA的两种角度：<font color=blue>最大投影方差，最小重构距离</font>；
  
    1. **最大投影方差**：数据在所找的主坐标轴投影的方差最大，最散
  
    <img src="F:\学习\编程语言相关\Python\img\PCA_1.png" alt="TU" style="zoom:50%;" />
  
    <center> PCA 最大投影方差图</center>
    - 先做特征空间的重构，找到一组正交的投影基$u_1\bot u_2$；
    
    - 找到$u_1$这种方向，使数据点在$u_1$上面的投影分布非常散，$u_1$称为主成分，若要降到$q$维，那就取$q$个主成分；
    
    - 步骤：
    
      1. 对每个数据点$x_i$, 先做去中心化：$x_i-\bar{x}$;
    
      2. 对每个数据点向$u_1$轴做投影
    
         - $J=\frac{1}{N}\sum_{i=1}^N{\left( \left( x_i-\bar{x} \right) ^Tu_1 \right) ^2},\,\,s.t.\,\,u_{1}^{T}u_1=1$
           $
           =\frac{1}{N}\sum_{i=1}^N{u_{1}^{T}\left( x_i-\bar{x} \right) \cdot \left( x_i-\bar{x} \right) ^Tu_1}
           $
           $
           =u_{1}^{T}\left( \frac{1}{N}\sum_{i=1}^N{\left( x_i-\bar{x} \right) \left( x_i-\bar{x} \right) ^T} \right) u_1
           $
           $=u_{1}^{T}Su_1$
           
           <font color=blue>先把样本$x_i$去中心化，然后再做投影，得到投影的坐标，再做平方</font>；
    
         - 原问题$\Rightarrow$$\left\{ \begin{array}{l}
           \hat{u}_1=arg\max u_{1}^{T}Su_1\\
             	s.t.\,\,u_{1}^{T}u_1=1\\
           \end{array} \right. $
    
         - 使用拉格朗日乘子：$L\left( u_1,\lambda \right) =u_{1}^{T}Su_1+\lambda \left( 1-u_{1}^{T}u_1 \right)$, $\frac{\partial L}{\partial u_1}=2Su_1-\lambda \cdot 2u_1=0$;
    
           解得：
           
           <font color=red>$Su_1=\lambda u_1$, $u_1$为特征向量，$\lambda$为特征值，$S$为样本的协方差。</font>
    
    
    2. **最小重构代价**：从投影中恢复数据的代价最小
       
       - 先做中心化；
       - 假设原坐标系有$p$维，重构坐标系为：$u_1,u_2,...,u_p$, 做降维之后的维度为$q$；
         - 重构之后样本点的坐标值：$x_i=\sum_{k=1}^p{\left( x_{i}^{T}u_k \right) u_k}$
         - 维度降为$q$之后的坐标值：$\hat{x}_i=\sum_{k=1}^q{\left( x_{i}^{T}u_k \right) u_k}$
       
       - 最小重构代价定义为两个坐标值的差值：
       
         $J=\frac{1}{N}\sum_{i=1}^N{\lVert x_i-\hat{x}_i \rVert ^2}$
         $
         =\frac{1}{N}\sum_{i=1}^N{\lVert \sum_{k=q+1}^p{\left( x_{i}^{T}u_k \right)} \rVert ^2}
         $
         $
         =\frac{1}{N}\sum_{i=1}^N{\sum_{k=q+1}^p{\left( x_{i}^{T}u_k \right) ^2}}
         $
       
         <font color=red>去中心化后：</font>
       
         $
         \frac{1}{N}\sum_{i=1}^N{\sum_{k=q+1}^p{\left( \left( x_i-\bar{x} \right) ^Tu_k \right) ^2}}
         $
         $
         =\sum_{i=1}^N{\sum_{k=q+1}^p{\frac{1}{N}\left( \left( x_i-\bar{x} \right) ^Tu_k \right) ^2}}
         $
         $
         =\sum_{k=q+1}^p{u_{k}^{T}Su_k},\ s.t.\ u_{k}^{T}u_k=1
         $
       
         <font color=red>原问题转化成：</font>
       
         $\left\{ \begin{array}{l}
         	\hat{u}_k=arg\min \sum_{k=q+1}^p{u_{k}^{T}Su_k}\\
         	s.t.\ u_{k}^{T}u_k=1\\
         \end{array} \right.$
    
    - <font color=blue>上面是对协方差矩阵$S$进行的特征值分解，下面通过对数据矩阵进行变换，用奇异值分解(Singular Value Decomposition, SVD):</font>
    
      - 对数据矩阵$X$进行SVD分解：
    
        $HX=U\varSigma V^T$, $HX$表示对数据矩阵做去中心化；
        $SVD$: $\left\{ \begin{array}{l}
        	U^TU=1\\
        	V^TV=VV^T=1\\
        \end{array} \right.$, 其中$\varSigma$为对角矩阵，$H$为中心矩阵：$\left\{ \begin{array}{l}
        	H^T=H\\
        	H^2=H\\
        \end{array} \right.$。
        
        对数据矩阵做$SVD$分解，带入到方差矩阵$S$：
        
        - $S=\frac{1}{N}X^THX$, $N$为常数，所以$S=X^THX$:
        
        - $
          S=X^THX
          $
          $
          =X^TH^THX
          $
          $
          =V\varSigma U^TU\varSigma V^T
          $
          $
          =V\varSigma ^2V^T
          $
        - <font color=red>从上面可以看出先对数据矩阵做$SVD$分解再代入到方差矩阵中，就相当于对方差矩阵做了奇异值分解；</font>
      
      - 对方差矩阵进行特征值分解：
        - $
          S=GKG^T
          $
          $
          G^TG=I,\ K=\text{diag}\left\{ k_1,...,k_p \right\} ,\ k_1\ge k_2\ge ...\ge k_p
          $
        - 可以看到特征值分解的结果与$SVD$分解是相同的。
      
      - <font color=red>不求样本的方差矩阵，对数据做中心化之后，对它进行奇异值分解可以得到相同的效果。</font>其中：$G=V,\ K=\varSigma ^2$。
      
      - 主坐标分析(Principle Coordinate Analysis, PCoA)
        - 对矩阵$T$进行奇异值分解，直接得到坐标值：$T=HXX^TH=U\varSigma V^TV\varSigma U^T=U\varSigma ^2U^T$;
        - <font color=blue>优点：当维度比较高的时候$S_{p\times p}$的计算量比较大，可以求$T_{N\times N}$；</font>
        - <font color=red>PCA得到的是方向（主成分），PCoA得到的是坐标值。</font>



## 2.3 概率模型估计(Probability model estimation)

- 假设训练数据由一个概率模型生成，由训练数据学习概率模型的结构和参数。<font color=blue>概率模型的结构类型，或者说概率模型的集合事先给定，而模型的具体结构和参数从数据中自动学习；</font>
- <font color=red>学习的目标是找到最后可能生成数据的结构和参数；</font>
- 概率模型包括混合模型、概率图模型等；
  - 概率图模型包括**有向图模型**和**无向图模型**；

- 概率模型表示为条件概率分布$P_{\theta}=\left( x|z \right)$；
  - 其中随机变量$x$表示观测数据，可以是连续变量也可以是离散变量；
  - 随机变量$z$表示隐式结构，是离散变量；<font color=blue>模型是混合模型，$z$表示成分的个数；模型是概率图模型，$z$表示图的结构；</font>
  - 随即变量$\theta$表示参数。

<p style="color:red;background-color:gray;text-align:center;">需要补充内容！</p>
#  三、 采样方法

## 3.1 MCMC (Markov Chain Monte Carlo)

<font color=red>随机采样方法，做一些复杂运算的<strong>近似求解</strong>。</font>

- ==Monte Carlo引入==

  - 最先得Monte Carlo方法都是为了求解一些不太好求解的<font color=red>求和</font>和<font color=red>求积分</font>问题，

    - 如积分：$\theta =\int_a^b{f\left( x \right) dx}$, $f(x)$的原函数不好求，可以用Monte Carlo方法求近似解；

    - 最简单是在$[a,b]$之间采一个点，如$x_0$，用$f(x_0)$代表$[a,b]$之间的值，$\Longrightarrow $<font color=red>$
      \theta \approx \left( b-a \right) f\left( x_0 \right) 
      $</font>;

    - 上面的假设太粗糙，可以用<font color=red>$n$个值</font>，$x_0,x_1,...,x_n$来代表，$\Longrightarrow $ <font color=red>$\theta \approx \frac{b-a}{n}\sum_{i=0}^{n-1}{f\left( x_i \right)}$</font>;

      这其中隐含一个假设：$x$在$[a,b]$之间均匀分布，<font color=red>但是绝大多数情况不是均匀分布，那么结果可能会差很远。</font>

    - <font color=blue>$\Longrightarrow $ 如果可以得到$x$在$[a,b]$的概率分布函数$p(x)$,</font> 

      - $\theta =\int_a^b{f\left( x \right) dx}=\int_a^b{\frac{f\left( x \right)}{p\left( x \right)}p\left( x \right) dx}\approx \frac{1}{n}\sum_{i=0}^{n-1}{\frac{f\left( x_i \right)}{p\left( x_i \right)}},\ x_i\sim p\left( x \right) $

        <p style="color:red;background-color:gray;text-align:center;">那么问题就变成了求解p(x)。</p>

    - <font color=red>问题：如何求出$x$的分布$p(x)$的对应的若干个样本。</font>

      - 如何求$p(x)$？

      - 求得$p(x)$后，如何基于概率分布去采样基于基于这个概率分布的样本集？

        1. 常见的均匀分布uniform(0, 1)：

           <font color=red>通过线性同余发生器生成(0, 1)之间的伪随机样本；</font>

        2. 其他的常见的分布：正态分布，$\tau$分布，$\varGamma $分布，$Beta$分布等：

           <font color=red>可以通过uniform(0, 1)的样本转换而得；</font>

        3. 不常见的分布：

           <font color=red>Rejection Sampling （拒绝采样）。</font>

           - $p(x)$太复杂不能直接采样，可以设定一个<font color=blue>程序可采样的分布$q(x)如高斯分布$</font>，然后按照一定的方法拒绝某些样本，以达到接近$p(x)$分布的目的，<font color=red>$q(x)$为proposal distribution，是方便采样的函数。</font>

             <img src=".\img\rejection sampling.png" alt="rejection sampling" style="zoom:67%;" />

             <center>图3.1 Rejection Sampling (k为一个常数)</center>

           - <font color=red>采样方法：</font>
           
             - firstly，采样得到$q(x)$的一个样本$z_0$;
           
             - secondly，从均匀分布$(0, kq(z_0))$中采样得到一个值u。
           
             - <font color=red>若u落入阴影中，拒绝本次采样，否则，接受$z_0$，</font>
             
               $\Longrightarrow$ 可得$z_0,z_1,...,z_{n-1}$, n个样本。
             
           - 最后的Monte Carlo方法结果为$\frac{1}{n}\sum_{i=0}^{n-1}{\frac{f\left( z_i \right)}{p\left( z_i \right)}}$，用$q(x)$模拟$p(x)$。
           
           - <font color=red>注：</font>
           
             - 对一些二维分布$p(x,y)$，有时只能得到$p(x|y)$或者$p(y|x)$，很难得到$p(x,y)$的形式，此时不能用Rejection Sampling；
             - 对一些高维的复杂非常见分布$p(x_1,x_2,...,x_n)$，找$q(x)和k$非常困难。

- ==Summary:== 要使用Monte Carlo 方法作为一个通用的采样模拟求和的方法，必须解决：<font color=red>如何方便求得各种复杂概率分布和对应的采样样本集的问题。</font>

  <p style="color:blue;background-color:red;text-align:center">Markov Chain可以解决这个问题！</p>



## 3.2 Markov Chain

<p style="color:blue;background-color:gray;text-align:center;"><srong>Markov Chain 在很多时间序列模型中得到广泛应用，RNN, HMM, MCMC。</srong></p>
<font color=red>用来求解复杂概率分布的对应的采样样本集的问题。</font>

- Markov Chain 假设某一时刻状态转移的概率只依赖于它的前一个时刻，

  <font color=red>$\Longrightarrow $</font> $P\left( X_{t+1}|X_0,...,X_{t-1},X_t \right) =P\left( X_{t+1}|X_t \right) $

1. **Markov Chain 状态转移矩阵**

   - 一个非周期的Markov Chain 状态转移矩阵P，并且它的任意两个状态是连通的，那么$P_{ij}^{n}$与$i$无关，有：
     1. $\underset{n\rightarrow \infty}{\lim}P_{ij}^{n}=\pi \left( j \right) $
     2. $\underset{n\rightarrow \infty}{\lim}P^n=\left( \begin{matrix}
        	\pi \left( 1 \right)&		\cdots&		\pi \left( j \right)&		\cdots\\
           	\vdots&		\vdots&		\vdots&		\vdots\\
           	\pi \left( 1 \right)&		\cdots&		\pi \left( j \right)&		\cdots\\
           	\vdots&		\vdots&		\vdots&		\vdots\\
        \end{matrix} \right) $
     3. $\pi \left( j \right) =\sum_{i=0}^{\infty}{\pi \left( i \right) P_{ij}}$
     4. $\pi$为$\pi P=\pi$的唯一非负解，$\pi =\left[ \pi \left( 1 \right) ,\pi \left( 2 \right) ,...,\pi \left( j \right) ... \right] ,\ \sum_{i=1}^{\infty}{\pi \left( i \right)}=1$
     5. <font color=red>注：</font>
        - 两个状态连通，任一个状态经过有限步可以到达其他的任一个；
        - $\pi$通常称为Markov Chain 的平稳分布。
          - <font color=red>平稳分布：</font>从任何初始状态出发，Markov Chain 的第n分布在$n\rightarrow \infty $时都会趋于不变分布。

2. **基于Markov Chain 的采样**

   <font color=red>如果我们得到了某个平稳分布对应的Markov Chain 状态转移矩阵，就很容易采样出这个平稳分布的样本集。</font>

   - 假设任意的初始概率分布为$\pi_i(0)$，经过第一轮状态转移后为$\pi_1{(x)},...,$ 第$i$轮的概率分布为$\pi_i(x)$。假设经过$n$轮后收敛到平稳分布$\pi(x)$，即
     $$
     \pi _n\left( x \right) =\pi _{n+1}\left( x \right) =\cdots \cdots =\pi \left( x \right)
     $$
     对每一个分布$\pi_i(x)$, $\pi _i\left( x \right) =\pi _{i-1}\left( x \right) P=\pi _{i-2}\left( x \right) P^2=\pi _0\left( x \right) P^i$；

   - 现在开始进行采样：
  - 基于初始任意简单概率分布$\pi_0(x)$采样得到状态值$x_0$；
     - 基于条件概率分布$P(x|x_0)$采样状态值$x_1,......$;
     - 到第$n$次，我们认为此时的采样集$(x_n,x_{n+1},x_{n+2},...)$为符合我们的平稳分布的对应样本集，可以来做Monte Carlo 求和。

- <font color=red>基于Markov Chain 的采样过程：</font>
  1. 输入Markov Chain 状态转移矩阵P，设定状态转移次数阈值$n_1$，需要的样本数$n_2$；
  2. 从<font color=red>从任意简单概率分布</font>采样得到初始状态$x_0$；
  3. for t=0 to $n_1+n_2-1$; 从条件概率分布$P(x|x_t)$中采样得到$x_{t+1}$，样本集$\left( x_{n1},\ x_{n1+1},\ ...\ ,\ x_{n1+n2-1} \right) $为平稳分布对应的样本集。

- ==Summary:== <font color=red>但是如何求平稳分布$\pi$对应的Markov Chain 状态转移矩阵$P$。</font>





## 3.3 MCMC采样和M-H采样

1. **Markov Chain 的细致平稳条件**

   - 非周期Markov Chain 的状态转移矩阵P和概率分布$\pi(x)$对所有的$i,j$满足：
     $$
     \pi \left( i \right) P\left( i,j \right) =\pi \left( j \right) P\left( j,i \right)
     $$
     <font color=red>称$\pi(x)$为状态转移矩阵P的平稳分布。</font>

   - <font color=blue>找到$\pi(x)$满足细致平稳分布的矩阵P，</font>

     $
     \Longrightarrow 
     $ $\sum_{i=1}^{\infty}{\pi \left( i \right) P\left( i,j \right) =\sum_{i=1}^{\infty}{\pi \left( j \right) P\left( j,i \right) =\pi \left( j \right) \sum_{i=1}^{\infty}{P\left( j,i \right)}=\pi \left( j \right)}}$

     $\Longrightarrow $  $\pi P=\pi$，Markov Chain收敛性。

   - <font color=red>但是：</font>
     - 从平稳细致条件很难找到合适的P；
     
     - 如我们的目标平稳分布为$\pi(x)$，随机找一个Markov Chain 转移矩阵Q，很难满足条件，即：
     
       <font color=red>$\pi \left( i \right) Q\left( i,j \right) \ne \pi \left( j \right) Q\left( j,i \right) 
       $</font>

2. **MCMC采样**

   <font color=red>引入$\alpha (i,j)$</font>, 使得$\pi \left( i \right) Q\left( i,j \right) \alpha \left( i,j \right) =\pi \left( j \right) Q\left( j,i \right) \alpha \left( j,i \right) $，其中$\left\{ \begin{array}{l}
   	\alpha \left( i,j \right) =\pi \left( j \right) Q\left( j,i \right)\\
   	\alpha \left( j,i \right) =\pi \left( i \right) Q\left( i,j \right)\\
   \end{array} \right. $， $\alpha (i,j)$表示接受率，范围：$[0,1]$。

   <font color=blue>所以，此时找到了$\pi (x)$对应的P</font>，$\Longrightarrow $ $P\left( i,j \right) =Q\left( i,j \right) \alpha \left( i,j \right) $。

   <p style="color:red;background-color:gray;text-align:center;">与Rejection Sampling类似，此处以一个常见的Q通过一定的拒绝接受概率得到P（目标转移矩阵）。</p>
- MCMC采样过程：
  
  - 任意选定Markov Chain 转移矩阵Q，平稳分布$\pi (x)$，状态转移次数阈值$n_1$，需要样本数$n_2$；
  
  - 从任意简单概率分布采样得到初始状态$x_0$；
  
  - $for \  t=0 \  to\  n_1 + n_2 -1$
  
    1. 从$Q(x|x_0)$中采样得$x_*$；
       2. 从均匀分布采样$u\sim uniform\left[ 0,1 \right] $；
       3. <font color=red>若$
          u<\alpha \left( x_t,u_* \right) =\pi \left( x_* \right) Q\left( x_*,x_t \right) ,$</font> 则接受转移$x_t\longmapsto x_*$, 即$x_{t+1}=x_*$；否则，拒绝，$x_{t+1}=x_t$；
       4. 得到$(x_{n1},x_{n1+1},...,x_{n1+n2-1})$为样本集。
  
    - <font color=red>注意：第3步中的$
         \alpha \left( x_t,x_* \right) 
         $可能非常小，如0.1，</font>导致大部分的采样值被拒绝转移，采样率低，<font color=blue>$
         \Longrightarrow 
         $ $M-H$采样。</font>
  
3. **Metropolis-Hastings**

   ==在MCMC的基础上对$\alpha (i,j)$进行改进==

   - <font color=blue>为什么要改进？</font>
     - 因为MCMC中$\alpha (i,j)$太小，导致采样率很低，但是可以将等式两边同时增大。

   $\Longrightarrow $  $\alpha (i,j)$改进：$\alpha \left( i,j \right) =\min \left\{ \frac{\pi \left( j \right) Q\left( j,i \right)}{\pi \left( i \right) Q\left( i,j \right)},\ 1 \right\} $;

   <font color=red>$M-H$采样过程和MCMC相同，只是$\alpha (i,j)$做了上面的变化。</font>

- ==Summary:==
  - <font color=blue>$M-H$采样完整地解决了Monte Carlo 方法需要的任意概率分布样本集的问题；</font>
  - <font color=red>现存的缺点:</font>
    - 由于$\frac{\pi \left( j \right) Q\left( j,i \right)}{\pi \left( i \right) Q\left( i,j \right)}$的存在，在<font color=red>计算高维时计算时间客观；</font>
    - <font color=blue>由于特征维度大，很难求出各个特征维度的联合分布，但可以方便地求出各个特征之间的条件概率分布。</font>

<p style="color=red;background-color:gray;text-align:center;">那么可不可以用条件概率分布方便地采样哪？</p>
## 3.4 Gibbs采样

- 改变平稳分布条件为条件概率；
- <font color=red>在处理高维特征是，MCMC一般指的是Gibbs; Gibbs一般需要两个维度。</font>

1. **重新寻找合适的平稳条件**

   - 原来的平稳条件：$\pi \left( i \right) P\left( i,j \right) =\pi \left( j \right) P\left( j,i \right) $，$\pi$为P的平稳分布；

   - 观察二维数据，设$\pi \left( x_1,x_2 \right) $为一个二维联合分布，第一个特征维度相同的两点：$A\left( x_{1}^{\left( 1 \right)},x_{2}^{\left( 1 \right)} \right) $和$B\left( x_{1}^{\left( 1 \right)},x_{2}^{\left( 2 \right)} \right)$

     - 对于A, B两点，下式子成立：

       - $
         \pi \left( x_{1}^{\left( 1 \right)},x_{2}^{\left( 1 \right)} \right) \pi \left( x_{2}^{\left( 2 \right)}|x_{1}^{\left( 1 \right)} \right) =\pi \left( x_{1}^{\left( 1 \right)} \right) \pi \left( x_{2}^{\left( 1 \right)}|x_{1}^{\left( 1 \right)} \right) \pi \left( x_{2}^{\left( 2 \right)}|x_{1}^{\left( 1 \right)} \right) 
         $
         $\pi \left( x_{1}^{\left( 1 \right)},x_{2}^{\left( 2 \right)} \right) \pi \left( x_{2}^{\left( 1 \right)}|x_{1}^{\left( 1 \right)} \right) =\pi \left( x_{1}^{\left( 1 \right)} \right) \pi \left( x_{2}^{\left( 2 \right)}|x_{1}^{\left( 1 \right)} \right) \pi \left( x_{2}^{\left( 1 \right)}|x_{1}^{\left( 1 \right)} \right) $

       - <font color=blue>上式右边相同</font>， 

         $\Longrightarrow$ $\pi \left( x_{1}^{\left( 1 \right)},x_{2}^{\left( 1 \right)} \right) \pi \left( x_{2}^{\left( 2 \right)}|x_{1}^{\left( 1 \right)} \right) =\pi \left( x_{1}^{\left( 1 \right)},x_{2}^{\left( 2 \right)} \right) \pi \left( x_{2}^{\left( 1 \right)}|x_{1}^{\left( 1 \right)} \right) $

         $\Longrightarrow$ $\pi \left( A \right) \pi \left( x_{2}^{\left( 2 \right)}|x_{1}^{\left( 1 \right)} \right) =\pi \left( B \right) \pi \left( x_{2}^{\left( 1 \right)}|x_{1}^{\left( 1 \right)} \right)$

     - <font color=blue>从上面的推导可以发现：</font>

       <img src=".\img\Gibbs.png" alt="Gibbs" style="zoom:67%;" />

       <center>图3.2 </center>
1. 在$x_1=x_{1}^{\left( 1 \right)}$这条线上，若使用概率分布$\pi \left( x_2|x_{1}^{\left( 1 \right)} \right)$作Markov Chain 的状态转移概率，<font color=red>那么任何两个点之间转移满足细致平稳条件。</font>
       2. 在$x_2=x_{2}^{\left( 1 \right)}$这条线上，若使用概率分布$\pi \left( x_1|x_{2}^{\left( 1 \right)} \right)$作Markov Chain 的状态转移概率，<font color=red>那么任何两个点之间转移满足细致平稳条件。</font>
       
- $\Longrightarrow $ $\pi \left( A \right) \pi \left( x_{1}^{\left( 2 \right)}|x_{2}^{\left( 1 \right)} \right) =\pi \left( C \right) \pi \left( x_{1}^{\left( 1 \right)}|x_{2}^{\left( 1 \right)} \right) $。
  
- <font color=red>可以构造平面任意两点的转移矩阵P：</font>
  
  $
       P\left( A\rightarrow B \right) =\pi \left( x_{2}^{\left( B \right)}|x_{1}^{\left( 1 \right)} \right) \,\,\,\,if\,\,\,\,x_{1}^{\left( A \right)}=x_{1}^{\left( B \right)}=x_{1}^{\left( 1 \right)}
       $
       $
       P\left( A\rightarrow C \right) =\pi \left( x_{1}^{\left( C \right)}|x_{2}^{\left( 1 \right)} \right) \,\,\,\,if\,\,\,\,x_{2}^{\left( A \right)}=x_{2}^{\left( C \right)}=x_{2}^{\left( 1 \right)}
       $
       $
       P\left( A\rightarrow D \right) =0\ \ \ \ \ \ \ \ \ \ \ \ \ else
       $
  
  $\Longrightarrow $ <font color =red>平面上任意两点$E、F$满足：</font> $\pi \left( E \right) P\left( E\rightarrow F \right) =\pi \left( F \right) P\left( F\rightarrow E \right) \ 
       $。
  
2. **二维Gibbs采样**

   1. 输入平稳分布$
      \pi \left( x_1,x_2 \right) 
      , n_1, n_2$;

   2. 随机初始化初始状态值$
      x_{1}^{\left( 0 \right)},x_{2}^{\left( 0 \right)}
      $；

   3. $for \ t=0 \ to \ n_1+n_2-1$

      - <font color=blue>从$
        P\left( x_2|x_{1}^{\left( t \right)} \right) 
        $中采样$
        x_{2}^{t+1}
        $</font>
        
      - <font color=blue>从$
        P\left( x_1|x_{2}^{\left( t \right)} \right) 
        $中采样$
        x_{1}^{t+1}
       $</font>
      
      - 样本集：$
        \left\{ \left( x_{1}^{\left( n1 \right)},x_{2}^{\left( n1 \right)} \right) ,\left( x_{1}^{\left( n1+1 \right)},x_{2}^{\left( n1+1 \right)} \right) ,...,\left( x_{1}^{\left( n1+n2-1 \right)},x_{2}^{\left( n1+n2-1 \right)} \right) \right\} 
       $
      
        <font color=blue>整个采样过程，坐标轮换：</font>
        $$
        \left( x_{1}^{\left( 1 \right)},x_{2}^{\left( 1 \right)} \right) \rightarrow \left( x_{1}^{\left( 1 \right)},x_{2}^{\left( 2 \right)} \right) \rightarrow \left( x_{1}^{\left( 2 \right)},x_{2}^{\left( 2 \right)} \right) \rightarrow ...\rightarrow \left( x_{1}^{\left( n1+n2-1 \right)},x_{2}^{\left( n1+n2-1 \right)} \right)
        $$
        

3. **多维Gibbs采样**
   
   - 改变二维中的步骤$3$将条件概率扩展到多维。
     1. 从$
        P\left( x_1|x_{2}^{\left( t \right)},x_{3}^{\left( t \right)},...,x_{n}^{\left( t \right)} \right) 
        $中采样得$x_1^{t+1}$;
     2. 从$
        P\left( x_2|x_{1}^{\left( t+1 \right)},x_{3}^{\left( t \right)},...,x_{n}^{\left( t \right)} \right) 
        $中采样得$x_2^{t+1}$;
     3. ………
     4. 从$
        P\left( x_j|x_{1}^{\left( t+1 \right)},x_{2}^{\left( t+1 \right)},...,x_{j-1}^{\left( t+1 \right)},x_{j+1}^{\left( t+1 \right)},...,x_{n}^{\left( t \right)} \right) 
        $采样得$x_n^{t+1}$;
     5. 从$
        P\left( x_n|x_{1}^{\left( t+1 \right)},x_{2}^{\left( t+1 \right)},...,x_{n-1}^{\left( t+1 \right)} \right) 
        $中采样得$x_n^{t+1}$。
   - <font color=red>与Lasso回归的坐标轴下降法<font color=blue>（固定$n-1$个特征，对某个特征求极值）</font>类似，</font>Gibbs是对某个特征进行采样。

#  四、 优化方法

## 4.1 梯度下降法(Gradient descent)

- 求解无约束问题（最优化问题）；

- 迭代算法，每一步需要求解目标函数的梯度向量；
  
  - $\underset{x\in \mathbb{R}^n}{\min}f\left( x \right)$, $f(x)$是$\mathbb{R}^n$上具有一阶连续偏导数的函数；
  
- 负梯度方向是函数值下降最快的方向，迭代的每一步以梯度的负方向更新$x$的值
  
- $f(x)$在$x^{\left( k \right)}$附近进行一阶泰勒展开：$f\left( x \right) =f\left( x^{\left( k \right)} \right) +g_{k}^{T}\left( x-x^{\left( k \right)} \right)$, $g_k=g\left( x^{\left( k \right)} \right) =\nabla f\left( x^{\left( k \right)} \right)$为$f(x)$在$x^{\left( k \right)}$的梯度。
  
- 主要可以分为三类：
  - <font color=red>批量梯度下降(Batch Gradient Descent)</font>
  - <font color=red>随机梯度下降(Stochastic Gradient Descent)</font>
  - <font color=red>小批量梯度下降法(Mini-Batch Gradient Descent)</font>

- 假设只有<font color=blue>一个特征</font>的线性回归来展开。

    - 此时线性回归的假设函数为：
    
      $$
      h_{\theta}\left( x^{\left( i \right)} \right) =\theta _1x^{\left( i \right)}+\theta _{\left( 0 \right)}
      $$
      $i=1,...,m$为样本数；
    
    - 相应的目标函数（代价函数）为：$J\left( \theta _0,\theta _1 \right) =\frac{1}{2m}\sum_{i=1}^m{\left( h_{\theta}\left( x^{\left( i \right)} \right) -y^{\left( i \right)} \right) ^2}$, $h_{\theta}\left( x^{\left( i \right)} \right)$为预测值，$y^{\left( i \right)}$为观测值。

1. **批量梯度下降(Batch Gradient Descent)**

   在每次迭代的时候用<font color=blue>所有的样本进行梯度更新；</font>

   - 对目标函数求导：$\frac{\partial J\left( \theta _0,\theta _1 \right)}{\partial \theta _j}=\frac{1}{m}\sum_{i=1}^m{\left( h_{\theta}\left( x^{\left( i \right)} \right) -y^{\left( i \right)} \right) x_{j}^{\left( i \right)}}$, $i=1,...,m$表示样本数；
     
   - 每次迭代对参数进行更新：

     $
    \theta _j:=\theta _j+\alpha p_j,\ 
     $
     $p_j=-\frac{\partial J}{\partial \theta _J}=-\frac{1}{m}\sum_{i=1}^m{\left( h_{\theta}\left( x^{\left( i \right)} \right) -y^{\left( i \right)} \right) x_{j}^{\left( i \right)}}$
     
     其中$p_i$表示梯度的反方向；<font color=red>这里有求和符号，表示对所有样本都计算。</font>
   
   - <font color=red>**优点**</font>
   
     - 一次迭代用了所有样本，可以利用矩阵进行操作，实现了并行；
     - <font color=blue>由全数据集确定的方向能够更好地代表样本总体</font>，从而更准确地向极值所在前进；
       - 对于凸集，一定能收敛到全局最小；
       - 对于非凸集，能收敛到局部最小。
   
   - <font color=red>**缺点**</font>
   
     - 样本总体m很大，计算会很慢；
   
     - 不能在线更新(online)。
   
2. **随机梯度下降法(Stochastic Gradient Descent)**

   <font color=red>与BGD相比，SGD每一迭代使用<font color=blue>一个样本</font>来对参数进行更新，加快训练速度。</font>

   - 对一个样本的目标函数：$J^{\left( i \right)}=\frac{1}{2}\left( h_{\theta}\left( x^{\left( i \right)} \right) -y^{\left( i \right)} \right) ^2$

     - 对目标函数求偏导：$\frac{\partial J^{\left( i \right)}\left( \theta _0,\theta _1 \right)}{\partial \theta _j}=\left( h_{\theta}\left( x^{\left( i \right)} \right) -y^{\left( i \right)} \right) x_{j}^{\left( i \right)}$；

     - 更新参数：

       $
       \theta _j:=\theta _j+\alpha p,\ 
       $
       $p=-\frac{\partial J^{\left( i \right)}\left( \theta _0,\theta _1 \right)}{\partial \theta _j}=\left( h_{\theta}\left( x^{\left( i \right)} \right) -y^{\left( i \right)} \right) x_{j}^{\left( i \right)}$

       $p$为梯度的反方向；<font color=red>这里只有一个样本的梯度，没有求和符号。</font>

   - <font color=red>**优点**</font>
     - 每轮迭代，只用一个数据进行更新梯度，加快了计算速度；
     - 可以做online learning。
   - <font color=red>**缺点**</font>
     - 只用一个数据更新方向，准确度下降；
     - 可能收敛到局部最优，<font color=blue>由于单个样本不能代表全体样本的趋势；</font>
     - 不易于进行并行实现。

3. 小批量梯度下降法(Mini-Batch Gradient Descent)**

   <font color=red>是对BGD和SGD的折中，每次采用$batch\_size$个样本进行参数更新。</font>

   $\theta _j:=\theta _j+\alpha p,\,\,$
   $\frac{\partial J\left( \theta _0,\theta _1 \right)}{\partial \theta _j}=\frac{1}{2b}\sum_{i=1}^b{\left( h_{\theta}\left( x^{\left( i \right)} \right) -y^{\left( i \right)} \right) x_{j}^{\left( i \right)}}$;

   - <font color=red>**优点**</font>

     - 可以用矩阵进行计算，每次在一个batch上优化神经网络参数并不会比单个数据慢太多；
     - 每次使用一个batch可以大大减少收敛所需迭代的次数；
     - 可以并行实现。

   - <font color=red>**缺点**</font>

     - $batch\_size$的选择不当可能会带来问题；

       - <font color=blue>合理范围内增大$batch\_size$的好处：</font>
         - 内存利用率提高；
         - 跑完一次epoch所需要的迭代次数减少，对相同的数据量的处理速度加快；
         - 在一定的范围内，$batch\_size$越大，其确定的下降方向越准，引起的训练震荡越小。

       - <font color=blue>盲目增大$batch\_size$的坏处：</font>
         - 内存利用率提高，但是内存容量可能撑不住；
         - $batch\_size$大到一定程度，确定的下降方向基本不再变化；
         - 跑完一次epoch所需的迭代次数减少，但是要达到相同的精度，所花费的时间大大增加了。

4. **梯度下降优化算法**

   <p style="color:blue;background-color:gray;text-align:center;"><strong>Momentum</strong></p>
   <img src=".\img\momentum.png" alt="momentum" style="zoom: 50%;" />


<center>图4.1 有无动量法得的对比</center>
   - 从图2 可以观察到：
  - 有momentum的振荡幅度变小，（由于前一个时刻的momentum的作用）；
    
   - 达到一定地点的时间变短。
   
   - <font color=red>优点：</font>

     ![momentum2](.\img\momentum2.png)

     <center>图4.2 优化示意图</center>
- <font color=blue>梯度负方向与Momentum同向，可以加速，在局部最小值时，可以冲出，如3点；反向时，若冲量值$>$梯度值，可以朝着冲量的方向继续走，甚至冲出山峰，跳出局部最小值，如4点；</font>
  
     - 可以加快速度。

> 1. 加权移动平均法
>
>    - $F_t=w_1A_{t-1}+w_2A_{t-2}+\cdots +w_nA_{t-n},\ \sum_{i=1}^n{w_i=1}$, 
>      - $w_i$是权重，$A_t$为$t$时刻的值。
>
> 2. 指数平均法
>
>    是对加权移动平均的改进；
>
>    - $F_{t+1}=\alpha x_t+\left( 1-\alpha \right) F_t$, 
>      - $x_t$是t时刻的真实值，$F_t$是t时刻的预测值。
>
> 3. 指数加权移动平均
>
>    - $V_t=\beta V_{t-1}+\left( 1-\beta \right) \theta _t$, 
>      - $V_t$是t时刻移动平均预测值，$V_{t-1}$是t-1时刻移动平均预测值（应该是前一个时间段的？），$\beta$表示对过去值得权重，$\theta_t$表示t时刻真实值。

<p style="color:red;background-color:gray">  =>   Momentum + GD</p>
- $\left\{ \begin{array}{l}
  	\theta :=\theta -v_t\\
    	v_t=\gamma v_{t-1}+\eta \nabla _{\theta}J\left( \theta \right)\\
  \end{array} \right.$
  - $v_{t-1}$表示上一次迭代的movement，$\nabla _{\theta}J\left( \theta \right)$表示当前的梯度；
  - 加上动量，有更快的收敛速度，减少了振荡幅度。

## 4.2 牛顿法(Newton method)

- 求解无约束问题；
- 迭代算法，每一步需求解目标函数的<font color=red>Hessian矩阵</font>
- $f(x)$在$x^k$附近展开：$f\left( x \right) =f\left( x^{\left( k \right)} \right) +g_{k}^{T}\left( x-x^{\left( k \right)} \right) +\frac{1}{2}\left( x-x^{\left( k \right)} \right) H\left( x^{\left( k \right)} \right) \left( x-x^{\left( k \right)} \right)$，$f(x)$有二阶连续偏导数，
  - $g_k=g\left( x^{\left( k \right)} \right) =\nabla f\left( x^{\left( k \right)} \right)$, 为$f(x)$的梯度向量在$x^{(k)}$的值；
  - $H(x^{(k)})$是$f(x)$的Hessian矩阵在$x^{(k)}$的值，$H\left( x \right) =\left[ \frac{\partial ^2f}{\partial x_i\partial x_j} \right] _{n\times n}$。

1. **牛顿法(Newton method)**

   牛顿法利用$\nabla f\left( x \right) =0$,

   -  假设$x^{(k+1)}$满足：<font color=red>$
     \nabla f\left( x^{\left( k+1 \right)} \right) =0
     $</font>, 根据上面的展开式可得<font color=blue>$
     \nabla f\left( x \right) =g_k+H_k\left( x-x^{\left( k \right)} \right) 
     $</font>;
     - $\Longrightarrow$ <font color=red>$
       g_k+H_k\left( x^{\left( k+1 \right)}-x^{\left( k \right)} \right) =0 ;$</font>
     - $\Longrightarrow$$x^{\left( k+1 \right)}=x^{\left( k \right)}-H_{k}^{-1}g_k\ ,$
       $
       \text{或 }x^{\left( k+1 \right)}=x^{\left( k \right)}+p_k,\ p_k=-H_{k}^{-1}g_k,\ H_kp_k=-g_k
       $

   - <font color=red>需要求$H_{k}^{-1}$, 计算复杂，需要进行改进。</font>

2. **拟牛顿法(Quasi-Newton method)**

   <font color=red>用$x=x^{k+1}$代入$
   \nabla f\left( x \right) =g_k+H_k\left( x-x^{\left( k \right)} \right) 
   $  </font>

   - <font color=blue>$   \Longrightarrow    $</font> $g_{k+1}-g_k=H_k\left( x^{\left( k+1 \right)}-x^{\left( k \right)} \right)$, 令$y_k=g_{k+1}-g_k,\ \delta _k=x^{\left( k+1 \right)}-x^{\left( k \right)}$，

   - <font color=blue>$   \Longrightarrow    $</font> <font color=red>拟牛顿条件：</font>$\left\{ \begin{array}{l}
     	y_k=H_k\delta _k\\
       	H_{k}^{-1}g_k=\delta _k\\
     \end{array} \right.$
     - <font color=blue>如果$H_k$是正定（$H_{k}^{-1}$也是正定），可以保证牛顿法的搜索方向$p_k$是下降方向；</font>
       - 方向：$p_k=-H_{k}^{-1}g_k$;
       - 迭代方程：$x=x^{\left( k \right)}+\lambda p_k=x^{\left( k \right)}-\lambda H_{k}^{-1}g_k$;
       - <font color=red>$\Longrightarrow$</font> $f\left( x \right) =f\left( x^{\left( k \right)} \right) -\lambda g_{k}^{T}H_{k}^{-1}g_k,$ $H_{k}^{-1}$正定；
         - $\Longrightarrow$ $g_{k}^{T}H_{k}^{-1}g_k>0$, $\lambda$是一个充分小的正数；
         - $\Longrightarrow$ $f\left( x \right) <f\left( x^{\left( k \right)} \right)$成立；
         - $\Longrightarrow$ $p_k$为下降方向。

   - 可以从拟牛顿条件得出：
     - <font color=blue>令$B_k$作$H_k$的近似，$
       B_{k+1}\delta _k=y_k
       $；</font>
     - <font color=blue>令$G_k$作$
       H_{k}^{-1}
       $的近似，$
       G_{k+1}y_k=\delta _k
       $；</font>
     - 通过这两种近似可以衍生出不同的拟牛顿法。

3. **DFP(Davidon-Fletcher-Powell)**

   - <font color=red>用$
     G_k
     $代替$     H_{k}^{-1}
     $</font>，$G_{k+1}=G_k+P_k+Q_k$，$P$和$Q$是两个附加项；
   - 代入式子，可得：
     - $G_{k+1}y_k=G_ky_k+P_ky_k+Q_ky_k$, 其中$
       \left\{ \begin{array}{l}
       	P_ky_k=\delta _k\\
       	Q_ky_k=-G_ky_k\\
       \end{array} \right. 
       $;
   
   - 取$\left\{ \begin{array}{l}
     	P_k=\frac{\delta _k\delta _{k}^{T}}{\delta _{k}^{T}y_k}\\
       	Q_k=-\frac{G_ky_ky_{k}^{T}G_k}{y_{k}^{T}G_ky_k}\\
     \end{array} \right. $
     - $\Longrightarrow $ $G_{k+1}=G_k+\frac{\delta _k\delta _{k}^{T}}{\delta _{k}^{T}y_k}-\frac{G_ky_ky_{k}^{T}G_k}{y_{k}^{T}G_ky_k}$ ，称为<font color=red>DFP算法。</font>

4. **BFGS(Broyden-Fletcher-Goldfarb-Shanno)**
   - <font color=red>用$B_k$代替$H_k$,</font> $B_{k+1}\delta _k=y_k,\ B_{k+1}=B_k+P_k+Q_k$;
   - 代入式子可得：
     - $B_{k+1}\delta _k=B_k\delta _k+P_k\delta _k+Q_k\delta _k$, 其中$\left\{ \begin{array}{l}
       	P_k\delta _k=y_k\\
         	Q_k\delta _k=-B_k\delta _k\\
       \end{array} \right. $;
     - $\Longrightarrow $ 取$B_{k+1}=B_k+\frac{y_ky_{k}^{T}}{y_{k}^{T}\delta _k}-\frac{B_k\delta _k\delta _{k}^{T}B_k}{\delta _{k}^{T}B_k\delta _k}$, 称为BFGS算法；

5. **L-BFGS**
   - <font color=blue>$L:limited-memory/storage$</font>
   - <font color=red>在BFGS中，要用一个$N\times N$的矩阵$B_k$，$N$很大时，存储$B_k$很耗费计算机资源；</font>
     - 解决：不存储$B_k$，存储序列${\delta_i}$和$y_i$ （固定存储最新的m个），需要$B_k$时，及利用序列进行计算。
     - <font color=red>空间复杂度变化：$O(N\times N)$$\longrightarrow $ $O(m\times N)$。</font>

# 五、 概率图模型

## 5.1 背景介绍

- 在概率论、统计学及机器学习中，概率图模型（Graphical Model）是用<font color=blue>图论方法</font>以<font color=red>表现数个独立随机变量之关联的一种建模法。</font>一个$p$个节点的图中，节点$i$对应一个随机变量，记为$X_i$。概率图模型被广泛地应用于<font color=blue>贝叶斯统计与机器学习</font>中。

- <font color=blue>概率</font><font color=red>图</font>

  - <font color=blue>概率：</font>将概率引入机器学习；

    - 关注的是<font color=red>高维的随机变量，$P(x_1,x_2,...,x_p)$：</font>
      - 边缘概率$p(x_i)$
      - 条件概率$p(y_j|x_i)$

    - ==一些法则==
      - <font color=red>Sum rule: </font>$
        p\left( x_1 \right) =\int{p\left( x_1,x_2 \right) dx_2}
        $；
      - <font color=red>Product rule: </font>$
        p\left( x_1,x_2 \right) =p\left( x_1 \right) p\left( x_2|x_1 \right) =p\left( x_2 \right) p\left( x_1|x_2 \right) 
        $；
      - <font color=red>Chain rule: </font>$
        p\left( x_1,x_2,...,x_p \right) =\prod_{i=1}^p{p\left( x_i|x_1,x_2,...,x_{i-1} \right)}
        $；
      - <font color=red>Bayesian rule: </font>$
        p\left( x_2|x_1 \right) =\frac{p\left( x_1,x_2 \right)}{p\left( x_1 \right)}=\frac{p\left( x_1,x_2 \right)}{\int{p\left( x_1,x_2 \right) dx_2}}=\frac{p\left( x_2 \right) p\left( x_1|x_2 \right)}{\int{p\left( x_2 \right) p\left( x_1|x_2 \right) dx_2}}
        $。

    - 高维随机变量的<font color=blue>困境</font>：<font color=red>维度高，计算复杂，</font>$p(x_1,x_2,...,x_p)$；

    - 针对困境进行简化：

      - 假设每个维度之间是相互独立的：$
        p\left( x_1,x_2,...,x_p \right) =\prod_{i=1}^p{p\left( x_i \right)}
        $；
        - Naive Bayes: $
          p\left( x|y \right) =\prod_{i=1}^p{p\left( x_i|y \right)}
          $；<font color=red>假设太强，要适当放松一点；</font>

      - <font color=red>放松假设——</font>Markov Property: $
        x_j\bot x_{i+1}|x_i,\ j<i
        $; <font color=red>依赖规则太简单；</font>
      - 从Markov Property 引申：<font color=blue>条件独立假设：</font>==概率图中的核心概念==；
        - $x_A\bot x_B|x_C$, $x_A,x_B,x_C$是集合，但是不相交。
          
  
- <font color=red>图：</font><font color=red>将图赋予概率的模型</font>，将概率嵌入到图中；
    - <font color=red>图的结点代表<font color=blue>随机变量</font>，有向图的边代表<font color=blue>条件概率</font>;</font>
  - 更加直观；
    - 便于构造更加高级的模型。

- ==概率图模型中主要讨论的3个问题：==
  - <font color=red>模型表示</font>，Representation，根据有向和无向分类（离散）；
    - 有向图：<font color=blue>Bayesian Network</font>
    - 无向图：<font color=blue>Markov Network</font>
    - 高斯图（连续）
      - 有向；
      - 无向；
  - <font color=red>推理算法</font>，Inference，<font color=blue>给定已知数据的情况下，求另外的一些概率分布；</font>
    - 精确推断
    - 近似推断
      - 确定性近似（变分推断）
      - 随机近似(MCMC)
  - <font color=red>学习问题</font>，Learning，<font color=blue>根据已有的数据将模型的参数学习出来；</font>
    - 参数学习
      - 完备数据
      - 隐变量，EM算法；
    - 结构学习，学习图的结构：结构+参数。

## 5.2 有向图Bayesian Network

- $P\left( x_1,x_2,...,x_p \right) =p\left( x_1 \right) \prod_{i=2}^p{p\left( x_i|x_1,x_2,...,x_{i-1} \right)}$

  条件独立性：$x_A\bot x_C|x_B$；

  因子分解：$p\left( x_1,x_2,...,x_p \right) =\prod_{i=1}^p{p\left( x_i|x_{pa\left( i \right)} \right) }$， $x_{pa}(i)$为$x_i$的父亲集合。

1. tail to tail

   <img src="F:\学习\编程语言相关\Python\img\tail to tail.png" alt="tail to tail" style="zoom:33%;" />

   <center>图5.1 tail to tail</center>
- $
     p\left( a,b,c \right) =p\left( a \right) p\left( b|a \right) p\left( c|a \right) 
     $, 使用因子分解；
     $
     p\left( a,b,c \right) =p\left( a \right) p\left( b|a \right) p\left( c|a,b \right) 
     $, 使用链式法则。
   - <font color=red>$
     \Longrightarrow 
     $</font> $
     p\left( c|a \right) =p\left( c|a,b \right) 
     $ <font color=red>$
     \Longrightarrow c\bot b|a
     $</font>
   - <font color=red>$
     \Longrightarrow 
     $</font> <font color=red>$
     p\left( c|a \right) p\left( b|a \right) =p\left( b,c|a \right) 
     $</font>
   - 若a被观测，则路径被阻塞，那么b与c就独立。
   
2. head to tail

   <img src="F:\学习\编程语言相关\Python\img\head to tail.png" alt="head to tail" style="zoom: 50%;" />

   <center>图5.2 head to tail</center>
- $
     a\bot c|b\Longleftrightarrow 
     $ <font color=red>若b被观测，则路径被阻塞。</font>

3. head to head

   

   <img src="F:\学习\编程语言相关\Python\img\head to head.png" alt="head to head" style="zoom: 67%;" />

   <center>图5.3 head to head</center>
- $
     p\left( a,b,c \right) =p\left( a \right) p\left( b \right) p\left( c|a,b \right) 
     $
     $
     =p\left( a \right) p\left( b|a \right) p\left( c|a,b \right) 
     $
   - <font color=red>$
     \Longrightarrow p\left( b \right) =p\left( b|a \right) \Longrightarrow a\bot b
     $</font>
   
- 默认情况下，$
     a\bot b
     $，路径阻塞；若c被观测，（a和b有关系），则路径是通的。， 

## 5.3 D-seperation

<img src="F:\学习\编程语言相关\Python\img\D划分.png" alt="D划分" style="zoom:50%;" />

<center>图5.4 D 划分</center>
- <font color=blue>我们要清楚，一个有向无环图是否暗示了一个特定的条件依赖表述<font color=red>$
  A\bot C|B
  $</font>。为解决这个问题，我们考虑从A中任意结点到C中任意结点的所有可能的路径。</font>如果它包含一个结点满足下面两个性质中的一个，我们说这样的路径被“阻隔”：
  1. 路径上的箭头以$head\  to \ tail$或$tail\ to \ tail$ 的方式交汇于这个结点，且这个结点在B中；
  2. 箭头以$head\ to \ head$ 的方式交汇于这个结点，且这个结点和它所有后继都不在B中；

- <font color=red>如果所有的路径都被“阻隔”，那么我们说B把A从C中D-划分开。</font>



## 5.4 Bayesian Network

- <font color=blue>从单一到混合；</font>
- <font color=blue>从有限到无限：</font>
  - 空间：随机变量的取值，离散$\rightarrow $连续；
  - 时间；

1. **单一**：Naive  Bayes, $\rightarrow $ $P\left( x|y \right) =\prod_{i=1}^p{p\left( x_i|y=1 \right)}$
2. **混合**：GMM: （用来做聚类）
3. **时间：**
   - Markov Chain
   - Gaussian Process（无限维高斯分布）

4. **连续：**Gaussian Bayesian Network

- <font color=red>混合模型+时间 </font> <font color=blue>$\rightarrow $</font> ==动态模型==
  - HMM（离散）
  - 线性动态系统，LDS，Kalman Filter （连续（高斯），线性）
  - 粒子滤波，Particle Filter （非高斯，非线性）

## 5.5 Markov Network

- 无向图表示的随机变量之间存在的<font color=blue>成对马尔科夫性(pairwise Markov property)、局部马尔科夫性(local Markov property)、全局马尔科夫性(global Markov property)</font>。

1. **成对马尔科夫性(pairwise Markov property)**

   - 设$u$和$v$是无向图$G$中任意两个没有边连接的结点，结点$u$和$v$分别对应随机变量$Y_u$和$Y_v$，其他所有结点为$O$，对应的随机变量组为$Y_O$；

   - 成对马尔科夫性指的是：<font color=red>给定随机变量组$Y_O$的条件下随机变量$Y_u$和$Y_v$是条件独立的</font>，即：
     $$
     P\left( Y_u,Y_v|Y_O \right) =P\left( Y_u|Y_O \right) P\left( Y_v|Y_O \right)
     $$

2. **局部马尔科夫性(local Markov property)**

   - 设$
     v\in V
     $是无向图$G$中任意一个结点，$W$是与$v$有边连接的所有结点，$O$是$v$和$W$之外的其他所有结点。$v$和$W$分别对应随机变量$Y_v$和$Y_W$，$O$表示的随机变量组是$Y_O$；

   - 局部马尔科夫性指的是：<font color=red>在给定随机变量组$Y_W$的条件下随机变量$Y_v$与随机变量组$Y_O$是独立的，</font>即：
     $$
     P\left( Y_v,Y_O|Y_W \right) =P\left( Y_v|Y_W \right) P\left( Y_O|Y_W \right) 
     $$
     <img src="F:\学习\编程语言相关\Python\img\局部马尔可夫性.png" alt="局部马尔可夫性" style="zoom:80%;" />
     
<center>图5.5 局部马尔科夫性</center>
3. **全局马尔科夫性(global Markov property)**

   - 设结点集合$A,B$是在无向图$G$中被结点集合$C$分开的任意结点集合，如图5.6。结点集合$A,B和C$所对应的随机变量组分别是$Y_A,Y_B,Y_C$。

   - 全局马尔科夫性是指：<font color=red>给定随机变量组$Y_C$条件下随机变量组$Y_A$和$Y_B$是条件独立的，</font>即：
     $$
     P\left( Y_A,Y_B|Y_C \right) =P\left( Y_A|Y_C \right) P\left( Y_B|Y_C \right)
     $$
     <img src="F:\学习\编程语言相关\Python\img\全局马尔科夫性.png" alt="全局马尔科夫性" style="zoom:80%;" />
   
     <center>图5.6 全局马尔科夫性</center>

- 条件独立性假设体现三个方面：1，2，3；
- 上面三个定义是等价的，$
  1\Leftrightarrow 2\Leftrightarrow 3
  $。

4. **概率图无向模型的因子分解**

   - <font color=red>团与最大团</font>
     
- 无向图$G$中任何两个结点均有边连接的结点子集称为团(clique)。若$c$是无向图$G$的一个团，并且不能再加进任何一个$G$的结点使其成为一个更大的团，则称此$C$为最大团(maximal clique)。
  
- <font color=blue>将概率无向图模型的联合概率分布表示为其最大团上的随机变量的函数的乘积形式的操作，称为<font color=red>概率无向图模型的因子分解(factorization)</font>。</font>
  
     - 给定概率无向图模型，设其无向图为$G$，$C$为$G$上的最大团，$Y_C$表示$C$对应的随机变量，那么概率无向图模型的联合概率分布$P(Y)$可写作图中所有最大团$C$上的函数$
       \varPsi _C\left( Y_C \right) 
       $的乘积形式，即：
       $$
       P\left( Y \right) =\frac{1}{Z}\prod_C{\varPsi _C\left( Y_C \right)}
       $$
       其中$Z$是规范化因子(normalization factor), $
       Z=\sum_Y{\prod_C{\varPsi _C\left( Y_C \right)}}
    $;
    
     - 函数$
       \varPsi _C\left( Y_C \right) 
       $称为<font color=red>势函数</font>，这里要求势函数是严格正的，通常定义为<font color=red>指数函数</font>：
       $$
    \varPsi _C\left( Y_C \right) =\exp \left\{ -E\left( Y_C \right) \right\}
   $$
      

## 5.6 常见的图模型

- 图模型(Graphical Model)
  - 有向图模型(Directed Graphical Model), <font color=red>适合为有单向依赖的数据建模；</font>
    - 静态贝叶斯网络 (Static Bayesian Network)
    - 动态贝叶斯网络(Dynamic Bayesian Network)
      - 隐马尔可夫模型(HMM)
      - 卡尔曼滤波器(Kalman Filter)
      - 粒子滤波器(Particle Filter)
  - 无向图模型(Undirected Graphical Model), <font color=red>适合实体间相互依赖的建模；</font>
    - 马尔科夫网络(Markov Network)
      - 吉布斯/玻尔兹曼机(Gibbs/Boltzmann machine)
      - 条件随机场(CRF)

# 六、 参数估计

## 6.1 频率派与贝叶斯派

1.  **频率派**
   - 认为世界是确定的，直接为事件本身建模，<font color=blue>也就是说事情在多次重复实验中趋于一个稳定的值$p$，这个值就是该事件的==概率==；</font>
   - 他们认为<font color=red>模型的参数$\theta$是一个未知的常量</font>，数据$X$为随机的变量，希望通过类似解方程组的方式从数据中求得该未知数；
     - 使用的参数估计方法：<font color=blue><strong>极大似然估计 (Maximum Likelihood Estimate, MLE)</strong></font>；
   - <font color=blue>频率派 <font color=red>$
     \Longrightarrow 
     $</font> 统计机器学习 <font color=red>$
     \Longrightarrow 
     $</font> 优化问题；</font>
2. **贝叶斯派**
   - 认为世界是不确定的，因获取的信息不同而异；<font color=red>假设对世界有一个预先的估计，然后</font><font color=blue>通过获取的信息来不断调整之前的预估计；</font>
   - 不试图对事件本身进行建模，而是从旁观者的角度来说，<font color=red>因此对于同一件事，不同的人掌握的==先验==不同，那么他们所认为的事件状态也不同；</font>
   - 贝叶斯派认为<font color=blue>模型的参数$\theta$是一个随机变量，<font color=red>$
     \theta \sim p\left( \theta \right) 
     $，$p(\theta)$为先验分布</font>，源自某种潜在分布，希望从数据中推知该分布；</font>
     - 使用的参数估计方法：<font color=blue><strong>极大后验概率估计 (Maximum A Posteriori estimation, MAP)</strong></font>；

> https://zhuanlan.zhihu.com/p/40024110

## 6.2 极大似然估计, MLE

- <font color=red>根据已知样本，希望通过调整模型参数来使得模型能够最大化样本情况出现的概率。</font>

1. **离散情况**

   - 总体$X$为离散型，其分布律为$
     P\left\{ X=x \right\} =P\left( x;\theta \right) 
     $, <font color=red>$
     \theta \in \varTheta 
     $的形式已知，且$\theta$为待估参数；</font>

     - $
       X_1,X_2,...,X_n
       $为来自样本$X$的样本，则$
       X_1,X_2,...,X_n
       $的联合概率分布为$
       \prod_{i=1}^n{p\left( x_i;\theta \right)}
       $；

     - 又设$
       x_1,x_2,...,x_n
       $是相应于样本$
       X_1,X_2,...,X_n
       $的一个样本值，已知样本$
       X_1,X_2,...,X_n
       $取到$
       x_1,x_2,...,x_n
       $的概率，即事件$
       \left\{ X_1=x_1,X_2=x_2,...,X_n=x_n \right\} 
       $发生的概率为：
       $$
       L\left( \theta \right) =L\left( x_1,x_2,...,x_n;\theta \right) =\prod_{i=1}^n{p\left( x_i;\theta \right) ,\ \theta \in \varTheta}
       $$
       
       - <font color=red>$L(\theta)$成为样本的似然函数。</font>
   
   - <font color=blue>引进的最大似然估计</font>，就是固定样本观察值$x_1,x_2,...,x_n$，在$\theta$取值的可能范围内挑选使似然函数$L(x_1,x_2,...,x_n;\theta)$达到最大的参数值$
     \hat{\theta}
     $，最为参数$\theta$的估计值，即取$
     \hat{\theta}
     $使得
     $$
     L\left( x_1,x_2,...,x_n;\theta \right) =\underset{\theta \in \varTheta}{\max}L\left( x_1,x_2,...,x_n \right) 
     $$
   
     - 常记为$
       \hat{\theta}\left( x_1,x_2,...,x_n \right) 
       $, 称为参数$\theta$的极大似然估计。
   
2. **连续情况**

   - $X$为连续值，概率密度$f(x;\theta)$，<font color=red>$x_1,x_2,...,x_n$的联合概率密度为 $
     \prod_{i=1}^n{f\left( x_i;\theta \right)}
     ;$</font> 似然函数为
     $$
     L\left( \theta \right) =L\left( x_1,x_2,...,x_n;\theta \right) =\prod_{i=1}^n{f\left( x_i;\theta \right)}
     $$

     - $
       \hat{\theta}\left( x_1,x_2,...,x_n \right) 
       $为$\theta$的最大似然估计值；
     - <font color=blue>$
       \frac{d}{d\theta}L\left( \theta \right) =0\ \text{或}\frac{d}{d\theta}\ln L\left( \theta \right) =0
       $
       $
       \Longrightarrow \ \text{求得}\theta 
       $</font>

## 6.3 极大后验概率估计, MAP

- <font color=red>根据已知样本，来通过调整模型参数使得模型能够产生该数据样本的概率最大，只不过对于模型参数有了一个先验假设，即模型参数可能满足某种分布，不再一味地依赖数据样例。</font>

- 贝叶斯定理：
  $$
  p\left( \theta |X \right) =\frac{p\left( X|\theta \right) p\text{(}\theta \text{)}}{p\left( X \right)}\propto p\left( X|\theta \right) p\left( \theta \right)
  $$

  - $p\left( \theta |X \right) 为后验概率，p\left( X|\theta \right) 为似然概率，p(\theta)为先验概率；$

- <font color=blue>取后验概率最大：</font>$
  \theta _{\max}=\underset{\theta}{arg\max}\ p\left( \theta |X \right) =\underset{\theta}{arg\max}\ p\left( X|\theta \right) p\left( \theta \right) 
  $；

## 6.4 MLE与MAP的联系与区别

- MLE对参数$\theta$的估计方法：
  $$
  \begin{aligned}
  \hat{\theta}_{MLE}&=\underset{\theta}{arg\max}\ L\left( \theta \right) \\
  &=\underset{\theta}{arg\max}\ \prod_{i=1}^n{p\left( x_i;\theta \right)}\\
  &=\underset{\theta}{arg\max}\ \log \prod_{i=1}^n{p\left( x_i;\theta \right)}\\
  &=\underset{\theta}{arg\max}\ \sum_{i=1}^n{\log p\left( x_i;\theta \right)}\\
  &=\underset{\theta}{arg\min}\ -\sum_{i=1}^n{\log p\left( x_i;\theta \right)}
  \end{aligned}
  $$
  
- MAP对参数$\theta$的估计方法：
  $$
  \begin{aligned}
  \hat{\theta}_{MAP}&=\underset{\theta}{arg\max}\ p\left( \theta |X \right) \\
  &=\underset{\theta}{arg\max}\ p\left( X|\theta \right) p\left( \theta \right) \\
  &=\underset{\theta}{arg\max}\ \log p\left( X|\theta \right) p\left( \theta \right)\\ 
  &=\underset{\theta}{arg\max}\ \log p\left( X|\theta \right) +\log p\left( \theta \right)\\ 
  &=\underset{\theta}{arg\min}\ -\log p\left( X|\theta \right) -\log p\left( \theta \right)
  \end{aligned}
  $$
  
- 对比上面两组公式：
  
  - MLE和MAP在优化时的<font color=red>不同在于MAP增加了一个先验项$-logp(\theta)$</font>；

## 6.5 EM算法

- EM算法是一种<font color=red>迭代算法</font>，1997年由Dempster 等人总结提出，<font color=blue>用于含有隐变量的概率模型参数的极大似然估计或极大后验估计</font>；EM算法的每次迭代分成两步：

  - E步，<font color=blue>求期望(expectation)</font>；

  - M步，<font color=blue>求极大(maximization)</font>；

    <font color=red>所以这个算法称为期望极大算法(expectation maximization algrithm)，简称EM算法。</font>

- 概率模型有时既含有观测变量(observable variable)，又含有<font color=red>隐变量或潜在变量(latent variable)</font>；
  - 如果概率模型的<font color=red>变量都是观测变量</font>，那么给定数据，可以直接用极大似然估计或极大后验估计估计模型参数；
  - 但<font color=red>含有隐变量</font>时，就不能用MLE和MAP，可以用EM算法，<font color=blue>EM算法就是含有隐变量的概率模型参数的极大似然估计法或极大后验估计法</font>；
  - <font color=blue>隐变量</font>是指不能被直接观察到的，但是对系统的状态和输出存在一定影响的一种东西。

1. **EM算法**

   - 一般地，用$Y$表示观测随机变量的数据，$Z$表示因随机变量的数据，$Y$和$Z$连在一起称为<font color=blue>完全数据(complete-data)</font>，观测数据$Y$又称为不完全数据(incomplete-data)；

     - 假设给定观测数据$Y$，其概率分布是$P(Y|\theta)$，其中$\theta$是需要估计得模型参数，<font color=blue>那么$Y$的似然函数是$P(Y|\theta)$，对数似然函数是$L(\theta)=logP(Y|\theta)$; </font>

     - 假设$Y$和$Z$的联合概率密度是$P(Y,Z|\theta)$，那么<font color=blue>完全数据的对数似然函数是$logP(Y,Z|\theta)$</font>；
   
   - <font color=red>EM算法通过迭代求$L(\theta)=logP(Y|\theta)$的极大似然估计</font>，每次迭代包含两步：求期望；求极大化。
   
   - EM算法：
   
     - 输入：观测变量数据$Y$，隐变量数据$Z$，联合概率分布$P(Y,Z|\theta)$，条件分布$P(Z|Y,\theta)$；
     - 输出：模型参数$\theta$；
   
     1. 选择参数的初值$\theta^{(0)}$，开始迭代；
   
     2. <font color=red>E步：</font>记$\theta^{(i)}$为第$i$次迭代参数$\theta$的估计值，在第$i+1$次迭代的E步，计算
        $$
        \begin{aligned}
        Q\left( \theta ;\theta ^{\left( i \right)} \right) &=E_Z\left[ \log P\left( Y,Z|\theta \right) |Y,\theta ^{\left( i \right)} \right] \\
        &=\sum_Z{\log P\left( Y,Z|\theta \right) P\left( Z|Y,\theta ^{\left( i \right)} \right)}
        \end{aligned}
        $$
        <font color=red>上面的$Q(\theta , \theta^{(i)})$函数是EM算法的核心，称为$Q$函数(Q function)。</font>
   
     3. <font color=red>M步：</font>求使$Q(\theta , \theta^{(i)})$极大化的$\theta$，确定第$i+1$次迭代的参数的估计值$\theta ^{(i+1)}$
        $$
        \theta ^{\left( i+1 \right)}=\underset{\theta}{arg\max}\ Q\left( \theta ,\theta ^{\left( i \right)} \right)
        $$
   
     4. 重复第2步和第3步，直到收敛。
   
     - <font color=red>$Q$函数：</font>完全数据的对数似然函数$logp(Y,Z|\theta)$关于在给定观测数据$Y$和当前参数$\theta$下对未观测数据$Z$的条件概率分布$P(Z|Y,\theta^{(i)})$的期望称为$Q$函数，即
       $$
       Q\left( \theta ;\theta ^{\left( i \right)} \right) =E_Z\left[ \log P\left( Y,Z|\theta \right) |Y,\theta ^{\left( i \right)} \right]
       $$
       
     - <font color=red>几点说明：</font>
     - 参数的初始值可以任意选择，但是要注意的是EM算法对初始值是敏感的；
       
       - $Q(\theta , \theta^{(i)})$中第一个变元是参数，第二个变元表示参数的当前估计值，<font color=red>每次迭代实际是在求$Q$函数及其极大。</font>
   
2. **EM算法的导出**

## 6.6 贝叶斯估计

- MLE和MAP都是<font color=red>点估计</font>，<font color=blue>是将使得产生训练样本的概率最大的参数作为这些参数的最佳估计。</font>贝叶斯估计是在MAP的基础上做进一步的拓展，此时<font color=red>不直接估计参数$w$的值，而是允许参数服从一定的概率分布；</font>

- 贝叶斯公式：
  $$
  p\left( \theta |X \right) =\frac{p\left( X|\theta \right) p\text{(}\theta \text{)}}{p\left( X \right)}
  $$

  - $p\left( \theta |X \right) 为后验(posterior)概率，p\left( X|\theta \right) 为似然概率，p(\theta)为先验(prior)概率；$
    - 先验概率：<font color=blue>没有掌握数据</font>的情况下参数的分布情况；
    - 后验概率：<font color=blue>掌握了一定量的数据</font>之后参数的分布情况。

  - 对于上面的贝叶斯公式，用$D$表示数据：
    $$
    p\left( \theta |D \right) =\frac{p\left( D|\theta \right) p\text{(}\theta \text{)}}{p\left( D \right)}
    $$

    - 式中分母$P(D)$可以看成一个归一化因子，其余的均是概率分布的函数，也就是说这里不能像极大似然估计一样将先验概率$p(\theta)$看成一个常数；

    - 用<font color=red>全概率公式</font>将分母展开：
      $$
      p\left( D \right) =\int_{\theta}{p\left( D|\theta \right) p\text{(}\theta \text{)}d\theta}
      $$

    - 将似然函数$
      p\left( D|\theta \right) =\prod_{i=1}^n{p\left( x_i|\theta \right)}
      $和上式代入贝叶斯公式可得：
      $$
      p\left( \theta |D \right) =\frac{\left( \prod_{i=1}^n{p\left( x_i|\theta \right)} \right) p\left( \theta \right)}{\int_{\theta}{\left( \prod_{i=1}^n{p\left( x_i|\theta \right)} \right) p\left( \theta \right) d\theta}}
      $$

  - 前面已经通过贝叶斯估计得到了后验概率$p(\theta|D)$，下面介绍一下应用；

    - 将问题形式化：<font color=red>已知数据$D=(x_1,x_2,...,x_n)$，预测新的数据x的值。</font>

      - 预测新的数据的值，就是能够在一直数据$D$的情况下，找到数据的数学期望：
        $$
        E\left( x|D \right) =\int_{\theta}{xp\left( x|D \right) dx}
        $$
        <font color=red>问题就是怎么求解$P(x|D)$?</font>

      - $P(x|D)$中的$x$的分布其实是与参数$\theta$有关的，参数$\theta$又是服从某种概率分布的，要对参数所有可能的情况进行考虑，可以得到：
        $$
        p\left( x|D \right) =\int_{\theta}{p\left( x,\theta |D \right) d\theta}
        $$

      - 运用基本的概率公式：$
        p\left( x,\theta |D \right) =p\left( x|\theta ,D \right) p\left( \theta |D \right) 
        $，<font color=blue>其中$
        p\left( x|\theta ,D \right) =p\left( x|\theta \right) 
        $，原因是从数据中得到的东西对于一个新数据来说就是参数，所以对$x$而言，$\theta$就是$D$，两者是同一条件，</font>可以得到：
        $$
        \begin{aligned}
        p\left( x|D \right) &=\int_{\theta}{p\left( x,\theta |D \right) d\theta}
        \\
        &=\int_{\theta}{p\left( x|\theta \right) p\left( \theta |D \right)}d\theta
        \end{aligned}
        $$

        - 上面的$p(x|\theta)$是已知的，$p(\theta|D)$在前面已经通过贝叶斯公式求出来。

  - ==贝叶斯估计中的一个困难==
    - 对于上面得到的式子$p(x|D)$，<font color=red>这里面困难的事参数是随机分布的，需要考虑每一个可能的参数情况然后进行积分，</font>这在数学上形式简单，但是需要很大的计算量；
    - 退而求其次，找一个效果差不多的后验概率，然后只将这个后验概率代入计算即可，那么怎样的后验概率和对所有可能的$\theta$积分情况差不多，通常的思想是<font color=blue><strong>找一个$\theta$能够最大化后验概率。</strong></font>就得到了极大后验概率估计。

- 贝叶斯估计与MAP
  - MAP是属于贝叶斯估计得一个trick，放弃了一点准确性，来极大地提升算法的性能；
  - MAP是一种点估计。
