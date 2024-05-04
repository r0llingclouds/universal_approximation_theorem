[Artificial neural networks](Artificial_neural_networks "wikilink") are
combinations of multiple simple mathematical functions that implement
more complicated functions from (typically) real-valued
[vectors](vector_(mathematics_and_physics) "wikilink") to real-valued
[vectors](vector_(mathematics_and_physics) "wikilink"). The spaces of
multivariate functions that can be implemented by a network are
determined by the structure of the network, the set of simple functions,
and its multiplicative parameters. A great deal of theoretical work has
gone into characterizing these function spaces.

In the [mathematical](mathematics "wikilink") theory of [artificial
neural networks](artificial_neural_networks "wikilink"), **universal
approximation theorems** are results[^1][^2] that put limits on what
neural networks can theoretically learn. Specifically, given an
algorithm that generates the networks within a class of functions, the
theorems establish the [density](dense_set "wikilink") of the generated
functions within a given function space of interest. Typically, these
results concern the approximation capabilities of the [feedforward
architecture](feedforward_neural_network "wikilink") on the space of
continuous functions between two [Euclidean
spaces](Euclidean_space "wikilink"), and the approximation is with
respect to the [compact convergence](compact_convergence "wikilink")
topology. What must be stressed, is that while some functions can be
arbitrarily well approximated in a region, the proofs do not apply
outside of the region, i.e. the approximated functions do not
[extrapolate](extrapolate "wikilink") outside of the region. That
applies for all non-periodic [activation
functions](activation_function "wikilink"), i.e. what's in practice used
and most proofs assume. In recent years neocortical pyramidal neurons
with oscillating activation function that can individually learn the XOR
function have been discovered in the human brain and oscillating
activation functions have been explored and shown to outperform popular
activation functions on a variety of benchmarks.[^3]

However, there are also a variety of results between [non-Euclidean
spaces](non-Euclidean_space "wikilink")[^4] and other commonly used
architectures and, more generally, algorithmically generated sets of
functions, such as the [convolutional neural
network](convolutional_neural_network "wikilink") (CNN)
architecture,[^5][^6] [radial basis
functions](radial_basis_functions "wikilink"),[^7] or neural networks
with specific properties.[^8][^9] Most universal approximation theorems
can be parsed into two classes. The first quantifies the approximation
capabilities of neural networks with an arbitrary number of artificial
neurons ("*arbitrary width*" case) and the second focuses on the case
with an arbitrary number of hidden layers, each containing a limited
number of artificial neurons ("*arbitrary depth*" case). In addition to
these two classes, there are also universal approximation theorems for
neural networks with bounded number of hidden layers and a limited
number of neurons in each layer ("*bounded depth and bounded width*"
case).

Universal approximation theorems imply that neural networks can
*represent* a wide variety of interesting functions with appropriate
weights. On the other hand, they typically do not provide a construction
for the weights, but merely state that such a construction is possible.
To construct the weight, neural networks are trained, and they may
converge on the correct weights, or not (i.e. get stuck in a local
optimum). If the network is too small (for the dimensions of input data)
then the universal approximation theorems do not apply, i.e. the
networks will not learn. What was once proven about the depth of a
network, i.e. a single hidden layer enough, only applies for one
dimension, in general such a network is too shallow. The width of a
network is also an important
[hyperparameter](hyperparameter "wikilink"). The choice of an
[activation function](activation_function "wikilink") is also important,
and some work, and proofs written about, assume e.g.
[ReLU](ReLU "wikilink") (or [sigmoid](sigmoid_function "wikilink"))
used, while some, such as a linear are known to *not* work (nor any
polynominal).

Neural networks with an unbounded (non-polynomial) activation function
have the universal approximation property.[^10]

The universal approximation property of width-bounded networks has been
studied as a *dual* of classical universal approximation results on
depth-bounded networks. For input dimension dx and output dimension dy
the minimum width required for the universal approximation of the
*[L<sup>p</sup>](Lp_space "wikilink")* functions is exactly max{dx + 1,
dy} (for a ReLU network). More generally this also holds if *both* ReLU
and a [threshold activation function](step_function "wikilink") are
used.[^11]

## History

One of the first versions of the *arbitrary width* case was proven by
[George Cybenko](George_Cybenko "wikilink") in 1989 for
[sigmoid](sigmoid_function "wikilink") activation functions.[^12] ,
Maxwell Stinchcombe, and [Halbert White](Halbert_White "wikilink")
showed in 1989 that multilayer [feed-forward
networks](feed-forward_network "wikilink") with as few as one hidden
layer are universal approximators.[^13] Hornik also showed in 1991[^14]
that it is not the specific choice of the activation function but rather
the multilayer feed-forward architecture itself that gives neural
networks the potential of being universal approximators. Moshe Leshno
*et al* in 1993[^15] and later Allan Pinkus in 1999[^16] showed that the
universal approximation property is equivalent to having a nonpolynomial
activation function. In 2022, Shen Zuowei, Haizhao Yang, and Shijun
Zhang[^17] obtained precise quantitative information on the depth and
width required to approximate a target function by deep and wide ReLU
neural networks.

The *arbitrary depth* case was also studied by a number of authors such
as Gustaf Gripenberg in 2003,[^18] Dmitry Yarotsky,[^19] Zhou Lu *et al*
in 2017,[^20] Boris Hanin and Mark Sellke in 2018[^21] who focused on
neural networks with ReLU activation function. In 2020, Patrick Kidger
and Terry Lyons[^22] extended those results to neural networks with
*general activation functions* such, e.g. tanh, GeLU, or Swish, and in
2022, their result was made quantitative by Leonie Papon and Anastasis
Kratsios[^23] who derived explicit depth estimates depending on the
regularity of the target function and of the activation function.

The question of minimal possible width for universality was first
studied in 2021, Park et al obtained the minimum width required for the
universal approximation of *[L<sup>p</sup>](Lp_space "wikilink")*
functions using feed-forward neural networks with
[ReLU](Rectifier_(neural_networks) "wikilink") as activation
functions.[^24] Similar results that can be directly applied to
[residual neural networks](residual_neural_network "wikilink") were also
obtained in the same year by Paulo Tabuada and Bahman Gharesifard using
[control-theoretic](Control_theory "wikilink") arguments.[^25][^26] In
2023, Cai[^27] obtained the optimal minimum width bound for the
universal approximation.

The bounded depth and bounded width case was first studied by Maiorov
and Pinkus in 1999.[^28] They showed that there exists an analytic
sigmoidal activation function such that two hidden layer neural networks
with bounded number of units in hidden layers are universal
approximators. Using algorithmic and computer programming techniques,
Guliyev and Ismailov constructed a smooth sigmoidal activation function
providing universal approximation property for two hidden layer
feedforward neural networks with less units in hidden layers.[^29] It
was constructively proved in 2018 paper[^30] that single hidden layer
networks with bounded width are still universal approximators for
univariate functions, but this property is no longer true for
multivariable functions.

Several extensions of the theorem exist, such as to discontinuous
activation functions,[^31] noncompact domains,[^32] certifiable
networks,[^33] random neural networks,[^34] and alternative network
architectures and topologies.[^35][^36]

In 2023 it was published that a three-layer neural network can
approximate any function (*continuous* and *discontinuous*),[^37]
however, the publication came without efficient learning algorithms for
approximating discontinuous functions.

## Arbitrary-width case

A spate of papers in the 1980s—1990s, from [George
Cybenko](George_Cybenko "wikilink") and etc, established several
universal approximation theorems for arbitrary width and bounded
depth.[^38][^39][^40][^41] See[^42][^43][^44] for reviews. The following
is the most often quoted:

Also, certain non-continuous activation functions can be used to
approximate a sigmoid function, which then allows the above theorem to
apply to those functions. For example, the [step
function](step_function "wikilink") works. In particular, this shows
that a [perceptron](perceptron "wikilink") network with a single
infinitely wide hidden layer can approximate arbitrary functions.

Such an *f* can also be approximated by a network of greater depth by
using the same construction for the first layer and approximating the
identity function with later layers.

The above proof has not specified how one might use a ramp function to
approximate arbitrary functions in $C_0(\R^n, \R)$. A sketch of the
proof is that one can first construct flat bump functions, intersect
them to obtain spherical bump functions that approximate the [Dirac
delta function](Dirac_delta_function "wikilink"), then use those to
approximate arbitrary functions in $C_0(\R^n, \R)$.[^45] The original
proofs, such as the one by Cybenko, use methods from functional
analysis, including the [Hahn-Banach](Hahn–Banach_theorem "wikilink")
and [Riesz representation](Riesz_representation_theorem "wikilink")
theorems.

The problem with polynomials may be removed by allowing the outputs of
the hidden layers to be multiplied together (the "pi-sigma networks"),
yielding the generalization:[^46]

## Arbitrary-depth case

The "dual" versions of the theorem consider networks of bounded width
and arbitrary depth. A variant of the universal approximation theorem
was proved for the arbitrary depth case by Zhou Lu et al. in 2017.[^47]
They showed that networks of width *n* + 4 with [ReLU](ReLU "wikilink")
activation functions can approximate any [Lebesgue-integrable
function](Lebesgue_integration "wikilink") on *n*-dimensional input
space with respect to [*L*<sup>1</sup> distance](L1_distance "wikilink")
if network depth is allowed to grow. It was also shown that if the width
was less than or equal to *n*, this general expressive power to
approximate any Lebesgue integrable function was lost. In the same
paper[^48] it was shown that [ReLU](ReLU "wikilink") networks with width
*n* + 1 were sufficient to approximate any
[continuous](continuous_function "wikilink") function of *n*-dimensional
input variables.[^49] The following refinement, specifies the optimal
minimum width for which such an approximation is possible and is due
to.[^50]

Together, the central result of[^51] yields the following universal
approximation theorem for networks with bounded width (see also[^52] for
the first result of this kind).

\left\\\hat{f}(x) - f(x)\right\\ \< \varepsilon. </math>

In other words, 𝒩 is [dense](dense_set "wikilink") in
*C*(𝒳; ℝ<sup>*D*</sup>) with respect to the topology of [uniform
convergence](uniform_convergence "wikilink").

*Quantitative refinement:* The number of layers and the width of each
layer required to approximate *f* to *ε* precision known;[^53] moreover,
the result hold true when 𝒳 and ℝ<sup>*D*</sup> are replaced with any
non-positively curved [Riemannian
manifold](Riemannian_manifold "wikilink"). }}

Certain necessary conditions for the bounded width, arbitrary depth case
have been established, but there is still a gap between the known
sufficient and necessary conditions.[^54][^55][^56]

## Bounded depth and bounded width case

The first result on approximation capabilities of neural networks with
bounded number of layers, each containing a limited number of artificial
neurons was obtained by Maiorov and Pinkus.[^57] Their remarkable result
revealed that such networks can be universal approximators and for
achieving this property two hidden layers are enough.

This is an existence result. It says that activation functions providing
universal approximation property for bounded depth bounded width
networks exist. Using certain algorithmic and computer programming
techniques, Guliyev and Ismailov efficiently constructed such activation
functions depending on a numerical parameter. The developed algorithm
allows one to compute the activation functions at any point of the real
axis instantly. For the algorithm and the corresponding computer code
see.[^58] The theoretical result can be formulated as follows.

Here “*σ*: ℝ → ℝ is *λ*-strictly increasing on some set *X*” means that
there exists a strictly increasing function *u*: *X* → ℝ such that
\|*σ*(*x*) − *u*(*x*)\| ≤ *λ* for all *x* ∈ *X*. Clearly, a
*λ*-increasing function behaves like a usual increasing function as *λ*
gets small. In the "*depth-width*" terminology, the above theorem says
that for certain activation functions depth-2 width-2 networks are
universal approximators for univariate functions and depth-3
width-(2*d* + 2) networks are universal approximators for *d*-variable
functions (*d* \> 1).

## Graph input

Achieving useful universal function approximation on graphs (or rather
on [graph isomorphism classes](Graph_isomorphism "wikilink")) has been a
longstanding problem. The popular graph convolutional neural networks
(GCNs or GNNs) can be made as discriminative as the Weisfeiler–Leman
graph isomorphism test.[^59] In 2020,[^60] a universal approximation
theorem result was established by Brüel-Gabrielsson, showing that graph
representation with certain injective properties is sufficient for
universal function approximation on bounded graphs and restricted
universal function approximation on unbounded graphs, with an
accompanying 𝒪(\|*V*\| ⋅ \|*E*\|)-runtime method that performed at state
of the art on a collection of benchmarks (where *V* and *E* are the sets
of nodes and edges of the graph respectively).

## See also

-   [Kolmogorov–Arnold representation
    theorem](Kolmogorov–Arnold_representation_theorem "wikilink")
-   [Representer theorem](Representer_theorem "wikilink")
-   [No free lunch theorem](No_free_lunch_theorem "wikilink")
-   [Stone–Weierstrass theorem](Stone–Weierstrass_theorem "wikilink")
-   [Fourier series](Fourier_series "wikilink")

## References

[Category:Theorems in
analysis](Category:Theorems_in_analysis "wikilink") [Category:Artificial
neural networks](Category:Artificial_neural_networks "wikilink")
[Category:Network
architecture](Category:Network_architecture "wikilink")
[Category:Networks](Category:Networks "wikilink")

[^1]: Hornik, Kurt; Stinchcombe, Maxwell; White, Halbert (January 1989). "Multilayer feedforward networks are universal approximators". Neural Networks. 2 (5): 359–366. doi:10.1016/0893-6080(89)90020-8.

[^2]: Balázs Csanád Csáji (2001) Approximation with Artificial Neural Networks; Faculty of Sciences; Eötvös Loránd University, Hungary

[^3]: Gidon, Albert; Zolnik, Timothy Adam; Fidzinski, Pawel; Bolduan, Felix; Papoutsi, Athanasia; Poirazi, Panayiota; Holtkamp, Martin; Vida, Imre; Larkum, Matthew Evan (3 January 2020). "Dendritic action potentials and computation in human layer 2/3 cortical neurons". Science. 367 (6473): 83–87. Bibcode:2020Sci...367...83G. doi:10.1126/science.aax6239. PMID 31896716.

[^4]: Kratsios, Anastasis; Bilokopytov, Eugene (2020). Non-Euclidean Universal Approximation (PDF). Advances in Neural Information Processing Systems. Vol. 33. Curran Associates.

[^5]: Zhou, Ding-Xuan (2020). "Universality of deep convolutional neural networks". Applied and Computational Harmonic Analysis. 48 (2): 787–794. arXiv:1805.10769. doi:10.1016/j.acha.2019.06.004. S2CID 44113176.

[^6]: Heinecke, Andreas; Ho, Jinn; Hwang, Wen-Liang (2020). "Refinement and Universal Approximation via Sparsely Connected ReLU Convolution Nets". IEEE Signal Processing Letters. 27: 1175–1179. Bibcode:2020ISPL...27.1175H. doi:10.1109/LSP.2020.3005051. S2CID 220669183.

[^7]: Park, J.; Sandberg, I. W. (1991). "Universal Approximation Using Radial-Basis-Function Networks". Neural Computation. 3 (2): 246–257. doi:10.1162/neco.1991.3.2.246. PMID 31167308. S2CID 34868087.

[^8]: Yarotsky, Dmitry (2021). "Universal Approximations of Invariant Maps by Neural Networks". Constructive Approximation. 55: 407–474. arXiv:1804.10306. doi:10.1007/s00365-021-09546-1. S2CID 13745401.

[^9]: Zakwan, Muhammad; d’Angelo, Massimiliano; Ferrari-Trecate, Giancarlo (2023). "Universal Approximation Property of Hamiltonian Deep Neural Networks". IEEE Control Systems Letters: 1. arXiv:2303.12147. doi:10.1109/LCSYS.2023.3288350. S2CID 257663609.

[^10]: Sonoda, Sho; Murata, Noboru (September 2017). "Neural Network with Unbounded Activation Functions is Universal Approximator". Applied and Computational Harmonic Analysis. 43 (2): 233–268. arXiv:1505.03654. doi:10.1016/j.acha.2015.12.005. S2CID 12149203.

[^11]: Park, Sejun; Yun, Chulhee; Lee, Jaeho; Shin, Jinwoo (2021). Minimum Width for Universal Approximation. International Conference on Learning Representations. arXiv:2006.08859.

[^12]: Cybenko, G. (1989). "Approximation by superpositions of a sigmoidal function". Mathematics of Control, Signals, and Systems. 2 (4): 303–314. CiteSeerX 10.1.1.441.7873. doi:10.1007/BF02551274. S2CID 3958369.

[^13]: Hornik, Kurt (1991). "Approximation capabilities of multilayer feedforward networks". Neural Networks. 4 (2): 251–257. doi:10.1016/0893-6080(91)90009-T. S2CID 7343126.

[^14]: Leshno, Moshe; Lin, Vladimir Ya.; Pinkus, Allan; Schocken, Shimon (January 1993). "Multilayer feedforward networks with a nonpolynomial activation function can approximate any function". Neural Networks. 6 (6): 861–867. doi:10.1016/S0893-6080(05)80131-5. S2CID 206089312.

[^15]: Pinkus, Allan (January 1999). "Approximation theory of the MLP model in neural networks". Acta Numerica. 8: 143–195. Bibcode:1999AcNum...8..143P. doi:10.1017/S0962492900002919. S2CID 16800260.

[^16]: Shen, Zuowei; Yang, Haizhao; Zhang, Shijun (January 2022). "Optimal approximation rate of ReLU networks in terms of width and depth". Journal de Mathématiques Pures et Appliquées. 157: 101–135. arXiv:2103.00502. doi:10.1016/j.matpur.2021.07.009. S2CID 232075797.

[^17]: Gripenberg, Gustaf (June 2003). "Approximation by neural networks with a bounded number of nodes at each level". Journal of Approximation Theory. 122 (2): 260–266. doi:10.1016/S0021-9045(03)00078-9.

[^18]: Yarotsky, Dmitry (October 2017). "Error bounds for approximations with deep ReLU networks". Neural Networks. 94: 103–114. arXiv:1610.01145. doi:10.1016/j.neunet.2017.07.002. PMID 28756334. S2CID 426133.

[^19]: Lu, Zhou; Pu, Hongming; Wang, Feicheng; Hu, Zhiqiang; Wang, Liwei (2017). "The Expressive Power of Neural Networks: A View from the Width". Advances in Neural Information Processing Systems. 30. Curran Associates: 6231–6239. arXiv:1709.02540.

[^20]: Hanin, Boris; Sellke, Mark (2018). "Approximating Continuous Functions by ReLU Nets of Minimal Width". arXiv:1710.11278 [stat.ML].

[^21]: Kidger, Patrick; Lyons, Terry (July 2020). Universal Approximation with Deep Narrow Networks. Conference on Learning Theory. arXiv:1905.08539.

[^22]: Kratsios, Anastasis; Papon, Léonie (2022). "Universal Approximation Theorems for Differentiable Geometric Deep Learning". Journal of Machine Learning Research. 23 (196): 1–73. arXiv:2101.05390.

[^23]: Tabuada, Paulo; Gharesifard, Bahman (2021). Universal approximation power of deep residual neural networks via nonlinear control theory. International Conference on Learning Representations. arXiv:2007.06007.

[^24]: Tabuada, Paulo; Gharesifard, Bahman (May 2023). "Universal Approximation Power of Deep Residual Neural Networks Through the Lens of Control". IEEE Transactions on Automatic Control. 68 (5): 2715–2728. doi:10.1109/TAC.2022.3190051. S2CID 250512115. (Erratum: doi:10.1109/TAC.2024.3390099)

[^25]: Cai, Yongqiang (2023-02-01). "Achieve the Minimum Width of Neural Networks for Universal Approximation". ICLR. arXiv:2209.11395.

[^26]: Maiorov, Vitaly; Pinkus, Allan (April 1999). "Lower bounds for approximation by MLP neural networks". Neurocomputing. 25 (1–3): 81–91. doi:10.1016/S0925-2312(98)00111-8.

[^27]: Guliyev, Namig; Ismailov, Vugar (November 2018). "Approximation capability of two hidden layer feedforward neural networks with fixed weights". Neurocomputing. 316: 262–269. arXiv:2101.09181. doi:10.1016/j.neucom.2018.07.075. S2CID 52285996.

[^28]: Guliyev, Namig; Ismailov, Vugar (February 2018). "On the approximation by single hidden layer feedforward neural networks with fixed weights". Neural Networks. 98: 296–304. arXiv:1708.06219. doi:10.1016/j.neunet.2017.12.007. PMID 29301110. S2CID 4932839.

[^29]: Baader, Maximilian; Mirman, Matthew; Vechev, Martin (2020). Universal Approximation with Certified Networks. ICLR.

[^30]: Gelenbe, Erol; Mao, Zhi Hong; Li, Yan D. (1999). "Function approximation with spiked random networks". IEEE Transactions on Neural Networks. 10 (1): 3–9. doi:10.1109/72.737488. PMID 18252498.

[^31]: Lin, Hongzhou; Jegelka, Stefanie (2018). ResNet with one-neuron hidden layers is a Universal Approximator. Advances in Neural Information Processing Systems. Vol. 30. Curran Associates. pp. 6169–6178.

[^32]: Ismailov, Vugar E. (July 2023). "A three layer neural network can represent any multivariate function". Journal of Mathematical Analysis and Applications. 523 (1): 127096. arXiv:2012.03016. doi:10.1016/j.jmaa.2023.127096. S2CID 265100963.

[^33]: Funahashi, Ken-Ichi (January 1989). "On the approximate realization of continuous mappings by neural networks". Neural Networks. 2 (3): 183–192. doi:10.1016/0893-6080(89)90003-8.

[^34]: Hornik, Kurt; Stinchcombe, Maxwell; White, Halbert (January 1989). "Multilayer feedforward networks are universal approximators". Neural Networks. 2 (5): 359–366. doi:10.1016/0893-6080(89)90020-8.

[^35]: Haykin, Simon (1998). *Neural Networks: A Comprehensive Foundation*, Volume 2, Prentice Hall.

[^36]: Hassoun, M. (1995) *Fundamentals of Artificial Neural Networks* MIT Press, p. 48

[^37]: Nielsen, Michael A. (2015). "Neural Networks and Deep Learning". {{cite journal}}: Cite journal requires |journal= (help)

[^38]: Hanin, B. (2018). Approximating Continuous Functions by ReLU Nets of Minimal Width. arXiv preprint arXiv:1710.11278.

[^39]: Park, Yun, Lee, Shin, Sejun, Chulhee, Jaeho, Jinwoo (2020-09-28). "Minimum Width for Universal Approximation". ICLR. arXiv:2006.08859.

[^40]: Shen, Zuowei; Yang, Haizhao; Zhang, Shijun (January 2022). "Optimal approximation rate of ReLU networks in terms of width and depth". Journal de Mathématiques Pures et Appliquées. 157: 101–135. arXiv:2103.00502. doi:10.1016/j.matpur.2021.07.009. S2CID 232075797.

[^41]: Lu, Jianfeng; Shen, Zuowei; Yang, Haizhao; Zhang, Shijun (January 2021). "Deep Network Approximation for Smooth Functions". SIAM Journal on Mathematical Analysis. 53 (5): 5465–5506. arXiv:2001.03040. doi:10.1137/20M134695X. S2CID 210116459.

[^42]: Juditsky, Anatoli B.; Lepski, Oleg V.; Tsybakov, Alexandre B. (2009-06-01). "Nonparametric estimation of composite functions". The Annals of Statistics. 37 (3). doi:10.1214/08-aos611. ISSN 0090-5364. S2CID 2471890.

[^43]: Poggio, Tomaso; Mhaskar, Hrushikesh; Rosasco, Lorenzo; Miranda, Brando; Liao, Qianli (2017-03-14). "Why and when can deep-but not shallow-networks avoid the curse of dimensionality: A review". International Journal of Automation and Computing. 14 (5): 503–519. arXiv:1611.00740. doi:10.1007/s11633-017-1054-2. ISSN 1476-8186. S2CID 15562587.

[^44]: Johnson, Jesse (2019). Deep, Skinny Neural Networks are not Universal Approximators. International Conference on Learning Representations.

[^45]: Xu, Keyulu; Hu, Weihua; Leskovec, Jure; Jegelka, Stefanie (2019). How Powerful are Graph Neural Networks?. International Conference on Learning Representations.

[^46]: Brüel-Gabrielsson, Rickard (2020). Universal Function Approximation on Graphs. Advances in Neural Information Processing Systems. Vol. 33. Curran Associates.
