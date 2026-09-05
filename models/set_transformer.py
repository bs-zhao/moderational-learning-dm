import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from mlp import MLP

class MultiHeadAttentionBlock(nn.Module):
    """Implements the MAB block from [1] which represents learnable cross-attention.

    In particular, it uses a so-called "Post-LN" transformer block [2] which applies
    layer norm following attention and following MLP. A "Pre-LN" transformer block
    can easily be implemented.

    [1] Lee, J., Lee, Y., Kim, J., Kosiorek, A., Choi, S., & Teh, Y. W. (2019).
        Set transformer: A framework for attention-based permutation-invariant neural networks.
        In International conference on machine learning (pp. 3744-3753). PMLR.

    [2] Xiong, R., Yang, Y., He, D., Zheng, K., Zheng, S., Xing, C., ... & Liu, T. (2020, November).
    On layer normalization in the transformer architecture.
    In International conference on machine learning (pp. 10524-10533). PMLR.
    """

    def __init__(
        self,
        embed_dim: int = 64,
        num_heads: int = 4,
        dropout: float = 0.05,
        mlp_depth: int = 2,
        mlp_width: int = 128,
        mlp_activation: str = "gelu",
        kernel_initializer: str = "lecun_normal",
        use_bias: bool = True,
        layer_norm: bool = True,
        **kwargs,
    ):
        """Creates a multi-head attention block which will typically be used as part of a
        set transformer architecture according to [1]. Corresponds to standard cross-attention.

        Parameters
        ----------
        embed_dim : int, optional
            Dimensionality of the embedding space, by default 64.
        num_heads : int, optional
            Number of attention heads, by default 4.
        dropout : float, optional
            Dropout rate applied to attention and MLP layers, by default 0.05.
        mlp_depth : int, optional
            Number of layers in the feedforward MLP block, by default 2.
        mlp_width : int, optional
            Width of each hidden layer in the MLP block, by default 128.
        mlp_activation : str, optional
            Activation function used in the MLP block, by default "gelu".
        kernel_initializer : str, optional
            Initializer for kernel weights, by default "lecun_normal".
        use_bias : bool, optional
            Whether to include bias terms in dense layers, by default True.
        layer_norm : bool, optional
            Whether to apply layer normalization before and after attention, by default True.
        **kwargs : dict
            Additional keyword arguments passed to the nn.Module base class.
        """

        super().__init__()

        self.input_projector = nn.Linear(embed_dim, embed_dim, bias=use_bias)
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            bias=use_bias,
            batch_first=True
        )
        self.ln_pre = nn.LayerNorm(embed_dim) if layer_norm else None

        widths = [embed_dim] + [mlp_width] * mlp_depth + [embed_dim]
        self.mlp = MLP(
            widths=widths,
            activation=mlp_activation,
            kernel_initializer=kernel_initializer,
            dropout=dropout,
            use_bias=use_bias,
            residual=False
        )
        self.output_projector = nn.Linear(embed_dim, embed_dim, bias=use_bias)
        self.ln_post = nn.LayerNorm(embed_dim) if layer_norm else None

    def forward(self, seq_x: Tensor, seq_y: Tensor, training: bool = False,
                key_padding_mask: Tensor = None, **kwargs) -> Tensor:
        """Performs the forward pass through the attention layer.

        Parameters
        ----------
        seq_x    : Tensor (e.g., np.ndarray, tf.Tensor, ...)
            Input of shape (batch_size, seq_size_x, input_dim), which will
            play the role of a query (Q).
        seq_y    : Tensor
            Input of shape (batch_size, seq_size_y, input_dim), which will
            play the role of key (K) and value (V).
        training : boolean, optional (default - True)
            Passed to the optional internal dropout and spectral normalization
            layers to distinguish between train and test time behavior.
        key_padding_mask : Tensor, optional
            Binary mask of shape (batch_size, seq_size_y) where padding positions are True.
        **kwargs : dict, optional (default - {})
            Additional keyword arguments passed to the internal attention layer,
            such as ``attention_mask`` or ``return_attention_scores``

        Returns
        -------
        out : Tensor
            Output of shape (batch_size, set_size_x, output_dim)
        """

        attn_out, _ = self.attention(seq_x, seq_y, seq_y, key_padding_mask=key_padding_mask)
        h = self.input_projector(seq_x) + attn_out

        if self.ln_pre is not None:
            h = self.ln_pre(h)

        out = h + self.output_projector(self.mlp(h))
        if self.ln_post is not None:
            out = self.ln_post(out)

        return out

class SetAttentionBlock(MultiHeadAttentionBlock):
    """Implements the SAB block from [1] which represents learnable self-attention.

    [1] Lee, J., Lee, Y., Kim, J., Kosiorek, A., Choi, S., & Teh, Y. W. (2019).
        Set transformer: A framework for attention-based permutation-invariant neural networks.
        In International conference on machine learning (pp. 3744-3753). PMLR.
    """

    def forward(self, input_set: Tensor, training: bool = False,
                key_padding_mask: Tensor = None, **kwargs) -> Tensor:
        """Performs the forward pass through the self-attention layer.

        Parameters
        ----------
        input_set  : Tensor (e.g., np.ndarray, tf.Tensor, ...)
            Input of shape (batch_size, set_size, input_dim)
        training   : boolean, optional (default - True)
            Passed to the optional internal dropout and spectral normalization
            layers to distinguish between train and test time behavior.
        key_padding_mask : Tensor, optional
            Binary mask of shape (batch_size, set_size) where padding positions are True.
        **kwargs   : dict, optional (default - {})
            Additional keyword arguments passed to the internal attention layer,
            such as ``attention_mask`` or ``return_attention_scores``

        Returns
        -------
        out : Tensor
            Output of shape (batch_size, set_size, output_dim)
        """

        return super().forward(input_set, input_set, training=training,
                             key_padding_mask=key_padding_mask, **kwargs)

class InducedSetAttentionBlock(nn.Module):
    """Implements the ISAB block from [1] which represents learnable self-attention specifically
    designed to deal with large sets via a learnable set of "inducing points".

    [1] Lee, J., Lee, Y., Kim, J., Kosiorek, A., Choi, S., & Teh, Y. W. (2019).
        Set transformer: A framework for attention-based permutation-invariant neural networks.
        In International conference on machine learning (pp. 3744-3753). PMLR.
    """

    def __init__(
        self,
        num_inducing_points: int,
        embed_dim: int = 64,
        num_heads: int = 4,
        dropout: float = 0.05,
        mlp_depth: int = 2,
        mlp_width: int = 128,
        mlp_activation: str = "gelu",
        kernel_initializer: str = "lecun_normal",
        use_bias: bool = True,
        layer_norm: bool = True,
        **kwargs,
    ):
        """Creates a self-attention attention block with inducing points (ISAB) which will typically
        be used as part of a set transformer architecture according to [1].

        Parameters
        ----------
        num_inducing_points : int, optional
            The number of inducing points for set-based dimensionality reduction.
        embed_dim : int, optional
            Dimensionality of the embedding space, by default 64.
        num_heads : int, optional
            Number of attention heads, by default 4.
        dropout : float, optional
            Dropout rate applied to attention and MLP layers, by default 0.05.
        mlp_depth : int, optional
            Number of layers in the feedforward MLP block, by default 2.
        mlp_width : int, optional
            Width of each hidden layer in the MLP block, by default 128.
        mlp_activation : str, optional
            Activation function used in the MLP block, by default "gelu".
        kernel_initializer : str, optional
            Initializer for kernel weights, by default "lecun_normal".
        use_bias : bool, optional
            Whether to include bias terms in dense layers, by default True.
        layer_norm : bool, optional
            Whether to apply layer normalization before and after attention, by default True.
        **kwargs : dict
            Additional keyword arguments passed to the nn.Module base class.
        """

        super().__init__()

        self.num_inducing_points = num_inducing_points
        self.inducing_points = nn.Parameter(
            torch.randn(self.num_inducing_points, embed_dim)
        )

        mab_kwargs = dict(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            mlp_depth=mlp_depth,
            mlp_width=mlp_width,
            mlp_activation=mlp_activation,
            kernel_initializer=kernel_initializer,
            use_bias=use_bias,
            layer_norm=layer_norm,
        )
        self.mab0 = MultiHeadAttentionBlock(**mab_kwargs)
        self.mab1 = MultiHeadAttentionBlock(**mab_kwargs)

        nn.init.xavier_uniform_(self.inducing_points)

    def forward(self, input_set: Tensor, training: bool = False,
                key_padding_mask: Tensor = None, **kwargs) -> Tensor:
        """Performs the forward pass through the self-attention layer.

        Parameters
        ----------
        input_set  : Tensor (e.g., np.ndarray, tf.Tensor, ...)
            Input of shape (batch_size, set_size, input_dim)
            Since this is self-attention, the input set is used
            as a query (Q), key (K), and value (V)
        training   : boolean, optional (default - True)
            Passed to the optional internal dropout and spectral normalization
            layers to distinguish between train and test time behavior.
        key_padding_mask : Tensor, optional
            Binary mask of shape (batch_size, set_size) where padding positions are True.
        **kwargs   : dict, optional (default - {})
            Additional keyword arguments passed to the internal attention layer,
            such as ``attention_mask`` or ``return_attention_scores``

        Returns
        -------
        out : Tensor
            Output of shape (batch_size, set_size, input_dim)
        """

        batch_size = input_set.shape[0]
        inducing_points_expanded = self.inducing_points.unsqueeze(0).expand(batch_size, -1, -1)

        h = self.mab0(inducing_points_expanded, input_set, training=training,
                     key_padding_mask=key_padding_mask, **kwargs)

        return self.mab1(input_set, h, training=training, **kwargs)

class PoolingByMultiHeadAttention(nn.Module):
    """Implements the pooling with multi-head attention (PMA) block from [1] which represents
    a permutation-invariant encoder for set-based inputs.

    [1] Lee, J., Lee, Y., Kim, J., Kosiorek, A., Choi, S., & Teh, Y. W. (2019).
        Set transformer: A framework for attention-based permutation-invariant neural networks.
        In International conference on machine learning (pp. 3744-3753). PMLR.

    Note: Currently works only on 3D inputs but can easily be expanded by changing
    the internals slightly or using ``torch.nn`` equivalent approaches.
    """

    def __init__(
        self,
        num_seeds: int = 1,
        embed_dim: int = 64,
        num_heads: int = 4,
        seed_dim: int = None,
        dropout: float = 0.05,
        mlp_depth: int = 2,
        mlp_width: int = 128,
        mlp_activation: str = "gelu",
        kernel_initializer: str = "lecun_normal",
        use_bias: bool = True,
        layer_norm: bool = True,
        **kwargs,
    ):
        """
        Creates a PoolingByMultiHeadAttention (PMA) block for permutation-invariant set encoding using
        multi-head attention pooling. Can also be used us a building block for `DeepSet` architectures.

        Parameters
        ----------
        num_seeds : int, optional (default=1)
            Number of seed vectors used for pooling. Acts as the number of summary outputs.
        embed_dim : int, optional (default=64)
            Dimensionality of the embedding space used in the attention mechanism.
        num_heads : int, optional (default=4)
            Number of attention heads in the multi-head attention block.
        seed_dim : int or None, optional (default=None)
            Dimensionality of each seed vector. If None, defaults to `embed_dim`.
        dropout : float, optional (default=0.05)
            Dropout rate applied to attention and MLP layers.
        mlp_depth : int, optional (default=2)
            Number of layers in the feedforward MLP applied before attention.
        mlp_width : int, optional (default=128)
            Number of units in each hidden layer of the MLP.
        mlp_activation : str, optional (default="gelu")
            Activation function used in the MLP.
        kernel_initializer : str, optional (default="lecun_normal")
            Initializer for kernel weights in dense layers.
        use_bias : bool, optional (default=True)
            Whether to include bias terms in dense layers.
        layer_norm : bool, optional (default=True)
            Whether to apply layer normalization before and after attention.
        **kwargs
            Additional keyword arguments passed to the nn.Module base class.
        """

        super().__init__()

        self.mab = MultiHeadAttentionBlock(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            mlp_depth=mlp_depth,
            mlp_width=mlp_width,
            mlp_activation=mlp_activation,
            kernel_initializer=kernel_initializer,
            use_bias=use_bias,
            layer_norm=layer_norm,
        )

        seed_dim = seed_dim if seed_dim is not None else embed_dim
        self.seed_vector = nn.Parameter(torch.randn(num_seeds, seed_dim))

        widths = [embed_dim] + [mlp_width] * mlp_depth + [embed_dim]
        self.feedforward = MLP(
            widths=widths,
            activation=mlp_activation,
            kernel_initializer=kernel_initializer,
            dropout=dropout,
            use_bias=use_bias,
            residual=False
        )

        nn.init.xavier_uniform_(self.seed_vector)

    def forward(self, input_set: Tensor, training: bool = False,
                key_padding_mask: Tensor = None, **kwargs) -> Tensor:
        """Performs the forward pass through the PMA block.

        Parameters
        ----------
        input_set  : Tensor (e.g., np.ndarray, tf.Tensor, ...)
            Input of shape (batch_size, set_size, input_dim)
            Since this is self-attention, the input set is used
            as a query (Q), key (K), and value (V)
        training   : boolean, optional (default - True)
            Passed to the optional internal dropout and spectral normalization
            layers to distinguish between train and test time behavior.
        key_padding_mask : Tensor, optional
            Binary mask of shape (batch_size, set_size) where padding positions are True.
        **kwargs   : dict, optional (default - {})
            Additional keyword arguments passed to the internal attention layer,
            such as ``attention_mask`` or ``return_attention_scores``

        Returns
        -------
        summary : Tensor
            Output of shape (batch_size, num_seeds * summary_dim)
        """

        set_x_transformed = self.feedforward(input_set)
        batch_size = input_set.shape[0]
        seed_tiled = self.seed_vector.unsqueeze(0).expand(batch_size, -1, -1)
        summaries = self.mab(seed_tiled, set_x_transformed, training=training,
                           key_padding_mask=key_padding_mask, **kwargs)
        return summaries.reshape(summaries.shape[0], -1)

class SetTransformer(nn.Module):
    """Implements the set transformer architecture from [1] which ultimately represents
    a learnable permutation-invariant function. Designed to naturally model interactions in
    the input set, which may be hard to capture with the simpler ``DeepSet`` architecture.

    [1] Lee, J., Lee, Y., Kim, J., Kosiorek, A., Choi, S., & Teh, Y. W. (2019).
        Set transformer: A framework for attention-based permutation-invariant neural networks.
        In International conference on machine learning (pp. 3744-3753). PMLR.

    Note: Currently works only on 3D inputs but can easily be expanded by using appropriate reshaping.
    """

    def __init__(
        self,
        input_dim: int = None,
        summary_dim: int = 16,
        embed_dims: tuple = (64, 64),
        num_heads: tuple = (4, 4),
        mlp_depths: tuple = (2, 2),
        mlp_widths: tuple = (128, 128),
        num_seeds: int = 1,
        dropout: float = 0.05,
        mlp_activation: str = "gelu",
        kernel_initializer: str = "lecun_normal",
        use_bias: bool = True,
        layer_norm: bool = True,
        num_inducing_points: int = None,
        seed_dim: int = None,
        **kwargs,
    ):
        """
        Creates a many-to-one permutation-invariant encoder, typically used as a summary net for embedding set-based,
        (i.e., exchangeable or IID) data. Use a TimeSeriesTransformer or a FusionTransformer for non-IID data.

        The number of multi-head attention block is inferred from the length of `embed_dims` tuple.

        Parameters
        ----------
        input_dim : int, optional (default - None)
            Dimensionality of the input features. If None, assumes input matches first embed_dim.
        summary_dim : int, optional (default - 16)
            Dimensionality of the final summary output.
        embed_dims  : tuple of int, optional (default - (64, 64))
            Dimensions of the keys, values, and queries for each attention block.
        num_heads   : tuple of int, optional (default - (4, 4))
            Number of attention heads for each embedding dimension.
        mlp_depths  : tuple of int, optional (default - (2, 2))
            Depth of the multi-layer perceptron (MLP) blocks for each component.
        mlp_widths  : tuple of int, optional (default - (128, 128))
            Width of each MLP layer in each block for each component.
        num_seeds   : int, optional (default - 1)
            Number of seeds to use for embedding.
        dropout     : float, optional (default - 0.05)
            Dropout rate applied to the attention and MLP layers. If set to None, no dropout is applied.
        mlp_activation : str, optional (default - 'gelu')
            Activation function used in the dense layers. Common choices include "relu", "elu", and "gelu".
        kernel_initializer : str, optional (default - 'lecun_normal')
            Initializer for the kernel weights matrix. Common choices include "glorot_uniform", "he_normal", etc.
        use_bias : bool, optional (default - True)
            Whether to include a bias term in the dense layers.
        layer_norm : bool, optional (default - True)
            Whether to apply layer normalization after the attention and MLP layers.
        num_inducing_points : int or None, optional (default - None)
            Number of inducing points used, if applicable. If set to None, this option is disabled.
        seed_dim : int or None, optional (default - None)
            Dimensionality of the seed embeddings. If None, it defaults to `summary_dim`.
        **kwargs : dict
            Additional keyword arguments passed to the base layer.
        """

        super().__init__()

        num_attention_layers = len(embed_dims)

        if input_dim is not None and input_dim != embed_dims[0]:
            self.input_projection = nn.Linear(input_dim, embed_dims[0])
        else:
            self.input_projection = None

        self.attention_blocks = nn.ModuleList()

        global_attention_settings = dict(
            dropout=dropout,
            mlp_activation=mlp_activation,
            kernel_initializer=kernel_initializer,
            use_bias=use_bias,
            layer_norm=layer_norm,
        )

        for i in range(num_attention_layers):
            layer_attention_settings = dict(
                num_heads=num_heads[i],
                embed_dim=embed_dims[i],
                mlp_depth=mlp_depths[i],
                mlp_width=mlp_widths[i],
            )

            if num_inducing_points is None:
                block = SetAttentionBlock(**(global_attention_settings | layer_attention_settings))
            else:
                isab_settings = dict(num_inducing_points=num_inducing_points)
                block = InducedSetAttentionBlock(
                    **(global_attention_settings | layer_attention_settings | isab_settings)
                )

            self.attention_blocks.append(block)

        pooling_settings = dict(
            num_heads=num_heads[-1],
            embed_dim=embed_dims[-1],
            mlp_depth=mlp_depths[-1],
            mlp_width=mlp_widths[-1],
            seed_dim=seed_dim,
            num_seeds=num_seeds,
        )
        self.pooling_by_attention = PoolingByMultiHeadAttention(**(global_attention_settings | pooling_settings))
        self.output_projector = nn.Linear(embed_dims[-1] * num_seeds, summary_dim)

        self.summary_dim = summary_dim

    def forward(self, input_set: Tensor, training: bool = False,
                key_padding_mask: Tensor = None, **kwargs) -> Tensor:
        """Compresses the input sequence into a summary vector of size `summary_dim`.

        Parameters
        ----------
        input_set  : Tensor (e.g., np.ndarray, tf.Tensor, ...)
            Input of shape (batch_size, set_size, input_dim)
        training   : boolean, optional (default - False)
            Passed to the optional internal dropout and spectral normalization
            layers to distinguish between train and test time behavior.
        key_padding_mask : Tensor, optional
            Binary mask of shape (batch_size, set_size) where padding positions are True.
        **kwargs   : dict, optional (default - {})
            Additional keyword arguments passed to the internal attention layer,
            such as ``attention_mask`` or ``return_attention_scores``

        Returns
        -------
        out : Tensor
            Output of shape (batch_size, summary_dim)
        """
        summary = input_set

        if self.input_projection is not None:
            summary = self.input_projection(summary)

        for block in self.attention_blocks:
            summary = block(summary, training=training,
                          key_padding_mask=key_padding_mask, **kwargs)
        summary = self.pooling_by_attention(summary, training=training,
                                          key_padding_mask=key_padding_mask, **kwargs)
        summary = self.output_projector(summary)
        return summary

if __name__ == "__main__":
    print("=== maskSet Transformer ===")

    embed_dim = 64
    batch_size = 3
    max_seq_len = 100

    seq_lengths = [80, 60, 95]

    input_set = torch.randn(batch_size, max_seq_len, embed_dim)

    key_padding_mask = torch.zeros(batch_size, max_seq_len, dtype=torch.bool)
    for i, length in enumerate(seq_lengths):
        if length < max_seq_len:
            key_padding_mask[i, length:] = True

    print(f": {input_set.shape}")
    print(f": {seq_lengths}")
    print(f"Mask: {key_padding_mask.shape}")
    print(f"mask: {key_padding_mask.sum(dim=1).tolist()}")

    model = SetTransformer(
        input_dim=embed_dim,
        embed_dims=(embed_dim, embed_dim),
        num_heads=(4, 4),
        mlp_depths=(2, 2),
        mlp_widths=(128, 128),
        seed_dim=embed_dim,
        num_seeds=1,
        summary_dim=32,
    )

    model.eval()
    with torch.no_grad():

        output = model(input_set, key_padding_mask=key_padding_mask)

    print(f": {output.shape}")
