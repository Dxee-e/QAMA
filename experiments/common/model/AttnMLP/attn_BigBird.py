"""change from https://github.com/google-research/bigbird/blob/master/bigbird/core/attention.py"""


import numpy as np
import torch
from torch import nn


MAX_SEQ_LEN = 64


import torch.nn.functional as F


def get_shape_list(tensor):
    """Returns a list of the shape of tensor."""
    return list(tensor.shape)


def get_single_block_row_attention(
    block_id,
    to_start_block_id,
    to_end_block_id,
    num_rand_blocks,
    window_block_left=1,
    window_block_right=1,
    global_block_left=1,
    global_block_right=1,
):
    """Generates random attention blocks for a single row."""

    # list of to_blocks from which to choose random attention
    to_block_list = np.arange(to_start_block_id, to_end_block_id, dtype=np.int32)
    # permute the blocks
    perm_block = np.random.permutation(to_block_list)

    # illegal blocks for the current block id, using window
    illegal_blocks = list(
        range(block_id - window_block_left, block_id + window_block_right + 1)
    )

    # Add blocks at the start and at the end
    illegal_blocks.extend(list(range(global_block_left)))
    illegal_blocks.extend(
        list(range(to_end_block_id - global_block_right, to_end_block_id))
    )

    # The second from_block cannot choose random attention on second last to_block
    if block_id == 1:
        illegal_blocks.append(to_end_block_id - 2)

    # The second last from_block cannot choose random attention on second to_block
    if block_id == to_end_block_id - 2:
        illegal_blocks.append(1)

    selected_random_blokcs = []

    for i in range(to_end_block_id - to_start_block_id):
        if perm_block[i] not in illegal_blocks:
            selected_random_blokcs.append(perm_block[i])
        if len(selected_random_blokcs) == num_rand_blocks:
            break
    return np.array(selected_random_blokcs, dtype=np.int32)


def bigbird_block_rand_mask_with_head(
    seq_length,
    block_size,
    num_heads,
    plan_from_length,
    plan_num_rand_blocks,
    window_block_left=1,
    window_block_right=1,
    global_block_top=1,
    global_block_bottom=1,
    global_block_left=1,
    global_block_right=1,
):
    """Create adjacency list of random attention.

    Args:
      seq_length: int. length of sequence.
      block_size: int. size of block in sequence.
      num_heads: int. total number of heads.
      plan_from_length: list. plan from lenght where num_rand are choosen from.
      plan_num_rand_blocks: list. number of rand blocks within the plan.
      window_block_left: int. number of blocks of window to left of a block.
      window_block_right: int. number of blocks of window to right of a block.
      global_block_top: int. number of blocks at the top.
      global_block_bottom: int. number of blocks at the bottom.
      global_block_left: int. Number of blocks globally used to the left.
      global_block_right: int. Number of blocks globally used to the right.

    Returns:
      adjacency list of size num_head where each element is of size
      from_seq_length//from_block_size-2 by num_rand_blocks
    """
    # Total number of blocks in the mmask
    num_blocks = seq_length // block_size
    # Number of blocks per plan
    plan_block_length = np.array(plan_from_length) // block_size
    # till when to follow plan
    max_plan_idx = plan_from_length.index(seq_length)
    # Random Attention adjajency list
    rand_attn = [
        np.zeros(
            (num_blocks, np.sum(plan_num_rand_blocks[: max_plan_idx + 1])),
            dtype=np.int32,
        )
        for i in range(num_heads)
    ]

    # We will go iteratively over the plan blocks and pick random number of
    # Attention blocks from the legally allowed blocks
    for plan_idx in range(max_plan_idx + 1):
        rnd_r_cnt = 0
        if plan_idx > 0:
            # set the row for all from_blocks starting from 0 to
            # plan_block_length[plan_idx-1]
            # column indx start fromm plan_block_length[plan_idx-1] and ends at
            # plan_block_length[plan_idx]
            if plan_num_rand_blocks[plan_idx] > 0:
                rnd_r_cnt = int(np.sum(plan_num_rand_blocks[:plan_idx]))
                curr_r_cnt = int(np.sum(plan_num_rand_blocks[: plan_idx + 1]))
                for blk_rw_idx in range(
                    global_block_top, plan_block_length[plan_idx - 1]
                ):
                    for h in range(num_heads):
                        # print("head", h, "blk_rw_idx", blk_rw_idx)
                        rand_attn[h][blk_rw_idx, rnd_r_cnt:curr_r_cnt] = (
                            get_single_block_row_attention(
                                block_id=blk_rw_idx,
                                to_start_block_id=plan_block_length[plan_idx - 1],
                                to_end_block_id=plan_block_length[plan_idx],
                                num_rand_blocks=plan_num_rand_blocks[plan_idx],
                                window_block_left=window_block_left,
                                window_block_right=window_block_right,
                                global_block_left=global_block_left,
                                global_block_right=global_block_right,
                            )
                        )

            for pl_id in range(plan_idx):
                if plan_num_rand_blocks[pl_id] == 0:
                    continue
                for blk_rw_idx in range(
                    plan_block_length[plan_idx - 1], plan_block_length[plan_idx]
                ):
                    rnd_r_cnt = 0
                    to_start_block_id = 0
                    if pl_id > 0:
                        rnd_r_cnt = int(np.sum(plan_num_rand_blocks[:pl_id]))
                        to_start_block_id = plan_block_length[pl_id - 1]
                    curr_r_cnt = int(np.sum(plan_num_rand_blocks[: pl_id + 1]))
                    for h in range(num_heads):
                        # print("head", h, "blk_rw_idx", blk_rw_idx)
                        rand_attn[h][blk_rw_idx, rnd_r_cnt:curr_r_cnt] = (
                            get_single_block_row_attention(
                                block_id=blk_rw_idx,
                                to_start_block_id=to_start_block_id,
                                to_end_block_id=plan_block_length[pl_id],
                                num_rand_blocks=plan_num_rand_blocks[pl_id],
                                window_block_left=window_block_left,
                                window_block_right=window_block_right,
                                global_block_left=global_block_left,
                                global_block_right=global_block_right,
                            )
                        )

        if plan_num_rand_blocks[plan_idx] == 0:
            continue
        # print("Start from here")
        curr_r_cnt = int(np.sum(plan_num_rand_blocks[: plan_idx + 1]))
        from_start_block_id = global_block_top
        to_start_block_id = 0
        if plan_idx > 0:
            rnd_r_cnt = int(np.sum(plan_num_rand_blocks[:plan_idx]))
            from_start_block_id = plan_block_length[plan_idx - 1]
            to_start_block_id = plan_block_length[plan_idx - 1]

        for blk_rw_idx in range(from_start_block_id, plan_block_length[plan_idx]):
            for h in range(num_heads):
                # print("head", h, "blk_rw_idx", blk_rw_idx)
                rand_attn[h][blk_rw_idx, rnd_r_cnt:curr_r_cnt] = (
                    get_single_block_row_attention(
                        block_id=blk_rw_idx,
                        to_start_block_id=to_start_block_id,
                        to_end_block_id=plan_block_length[plan_idx],
                        num_rand_blocks=plan_num_rand_blocks[plan_idx],
                        window_block_left=window_block_left,
                        window_block_right=window_block_right,
                        global_block_left=global_block_left,
                        global_block_right=global_block_right,
                    )
                )

    for nh in range(num_heads):
        rand_attn[nh] = rand_attn[nh][
            global_block_top : num_blocks - global_block_bottom, :
        ]
    return rand_attn


def get_rand_attn_plan(from_seq_length, from_block_size, num_rand_blocks):
    """Gives the plan of where to put random attention.

    Args:
      from_seq_length: int. length of from sequence.
      from_block_size: int. size of block in from sequence.
      num_rand_blocks: int. Number of random chunks per row.

    Returns:
      plan_from_length: ending location of from block
      plan_num_rand_blocks: number of random ending location for each block
    """
    # general plan
    plan_from_length = []
    plan_num_rand_blocks = []
    if (2 * num_rand_blocks + 5) < (from_seq_length // from_block_size):
        plan_from_length.append(int((2 * num_rand_blocks + 5) * from_block_size))
        plan_num_rand_blocks.append(num_rand_blocks)
        plan_from_length.append(from_seq_length)
        plan_num_rand_blocks.append(0)
    elif (num_rand_blocks + 5) < (from_seq_length // from_block_size):
        plan_from_length.append(int((num_rand_blocks + 5) * from_block_size))
        plan_num_rand_blocks.append(num_rand_blocks // 2)
        plan_from_length.append(from_seq_length)
        plan_num_rand_blocks.append(num_rand_blocks - (num_rand_blocks // 2))
    else:
        plan_from_length.append(from_seq_length)
        plan_num_rand_blocks.append(num_rand_blocks)

    return plan_from_length, plan_num_rand_blocks


def bigbird_block_rand_mask(
    from_seq_length,
    to_seq_length,
    from_block_size,
    to_block_size,
    num_rand_blocks,
    last_idx=-1,
):
    """Create adjacency list of random attention.

    Args:
      from_seq_length: int. length of from sequence.
      to_seq_length: int. length of to sequence.
      from_block_size: int. size of block in from sequence.
      to_block_size: int. size of block in to sequence.
      num_rand_blocks: int. Number of random chunks per row.
      last_idx: if -1 then num_rand_blocks blocks chosen anywhere in to sequence,
        if positive then num_rand_blocks blocks choosen only upto last_idx.

    Returns:
      adjacency list of size from_seq_length//from_block_size-2 by num_rand_blocks
    """
    rand_attn = np.zeros(
        (from_seq_length // from_block_size - 2, num_rand_blocks), dtype=np.int32
    )
    middle_seq = np.arange(1, to_seq_length // to_block_size - 1, dtype=np.int32)
    last = to_seq_length // to_block_size - 1
    if last_idx > (2 * to_block_size):
        last = (last_idx // to_block_size) - 1

    r = num_rand_blocks  # shorthand
    for i in range(1, from_seq_length // from_block_size - 1):
        start = i - 2
        end = i
        if i == 1:
            rand_attn[i - 1, :] = np.random.permutation(middle_seq[2:last])[:r]
        elif i == 2:
            rand_attn[i - 1, :] = np.random.permutation(middle_seq[3:last])[:r]
        elif i == from_seq_length // from_block_size - 3:
            rand_attn[i - 1, :] = np.random.permutation(middle_seq[:last])[:r]
            # Missing -3: should have been sliced till last-3
        elif i == from_seq_length // from_block_size - 2:
            rand_attn[i - 1, :] = np.random.permutation(middle_seq[:last])[:r]
            # Missing -4: should have been sliced till last-4
        else:
            if start > last:
                start = last
                rand_attn[i - 1, :] = np.random.permutation(middle_seq[:start])[:r]
            elif (end + 1) == last:
                rand_attn[i - 1, :] = np.random.permutation(middle_seq[:start])[:r]
            else:
                rand_attn[i - 1, :] = np.random.permutation(
                    np.concatenate((middle_seq[:start], middle_seq[end + 1 : last]))
                )[:r]
    return rand_attn


def full_bigbird_mask(
    from_seq_length, to_seq_length, from_block_size, to_block_size, rand_attn
):
    """Calculate BigBird attention pattern as a full dense matrix.

    Args:
      from_seq_length: int. length of from sequence.
      to_seq_length: int. length of to sequence.
      from_block_size: int. size of block in from sequence.
      to_block_size: int. size of block in to sequence.
      rand_attn: adjajency matrix for random attention.

    Returns:
      attention mask matrix of shape [from_seq_length, to_seq_length]
    """

    attn_mask = np.zeros((MAX_SEQ_LEN, MAX_SEQ_LEN), dtype=np.int32)
    for i in range(1, (MAX_SEQ_LEN // from_block_size) - 1):
        attn_mask[
            (i) * from_block_size : (i + 1) * from_block_size,
            (i - 1) * to_block_size : (i + 2) * to_block_size,
        ] = 1
        for j in rand_attn[i - 1, :]:
            attn_mask[
                i * from_block_size : (i + 1) * from_block_size,
                j * to_block_size : (j + 1) * to_block_size,
            ] = 1

    attn_mask[:from_block_size, :] = 1
    attn_mask[:, :to_block_size] = 1
    attn_mask[:, -to_block_size:] = 1
    attn_mask[-from_block_size:, :] = 1
    clipped_attn_mask = attn_mask[:from_seq_length, :to_seq_length]
    return np.array(clipped_attn_mask, dtype=bool)


def create_rand_mask_from_inputs(
    from_blocked_mask,
    to_blocked_mask,
    rand_attn,
    num_attention_heads,
    num_rand_blocks,
    from_seq_length,
    from_block_size,
):
    """Create 4D attention mask from a 3D tensor mask.

    Args:
      from_blocked_mask: 3D Tensor of shape [batch_size,
        from_seq_length//from_block_size, from_block_size].
      to_blocked_mask: 3D Tensor of shape [batch_size,
        to_seq_length//to_block_size, to_block_size].
      rand_attn: [batch_size, num_attention_heads,
        from_seq_length//from_block_size-2, num_rand_blocks]
      num_attention_heads: int. Number of attention heads.
      num_rand_blocks: int. Number of random chunks per row.
      from_seq_length: int. length of from sequence.
      from_block_size: int. size of block in from sequence.

    Returns:
      float Tensor of shape [batch_size, num_attention_heads,
                             from_seq_length//from_block_size-2,
                             from_block_size, num_rand_blocks*to_block_size].
    """
    num_windows = from_seq_length // from_block_size - 2
    # rand_mask = tf.reshape(
    #     tf.gather(to_blocked_mask, rand_attn, batch_dims=1),
    #     [-1, num_attention_heads, num_windows, num_rand_blocks * from_block_size],
    # )
    # rand_mask = tf.einsum("BLQ,BHLK->BHLQK", from_blocked_mask[:, 1:-1], rand_mask)
    # Implementing gather with batch_dims=1 in PyTorch
    batch_size = from_blocked_mask.shape[0]
    rand_mask = torch.stack([torch.gather(to_blocked_mask[i], dim=0, index=rand_attn[i].long()) for i in range(batch_size)], dim=0)
    rand_mask = rand_mask.reshape(batch_size, num_attention_heads, num_windows, num_rand_blocks * from_block_size)

    # Implementing einsum "BLQ,BHLK->BHLQK" in PyTorch
    from_blocked_mask_sliced = from_blocked_mask[:, 1:-1]
    batch_size, num_blocks, block_size = from_blocked_mask_sliced.shape
    _, num_heads, _, rand_mask_width = rand_mask.shape

    from_blocked_mask_reshaped = from_blocked_mask_sliced.unsqueeze(1).unsqueeze(2).expand(batch_size, num_heads, num_blocks, block_size)
    rand_mask_reshaped = rand_mask.unsqueeze(3).expand(batch_size, num_heads, num_blocks, block_size, rand_mask_width)

    rand_mask = from_blocked_mask_reshaped * rand_mask_reshaped
    return rand_mask


def create_band_mask_from_inputs(from_blocked_mask, to_blocked_mask):
    """Create 4D attention mask from a 3D blocked tensor mask.

    Args:
      from_blocked_mask: 3D Tensor of shape [batch_size,
        from_seq_length//from_block_size, from_block_size].
      to_blocked_mask: 3D Tensor of shape [batch_size,
        to_seq_length//to_block_size, to_block_size].

    Returns:
      float Tensor of shape [batch_size, 1, from_seq_length//from_block_size-4,
                             from_block_size,  3*to_block_size].
    """
    # exp_blocked_to_pad = tf.concat(
    #     [to_blocked_mask[:, 1:-3], to_blocked_mask[:, 2:-2], to_blocked_mask[:, 3:-1]],
    #     2,
    # )
    exp_blocked_to_pad = torch.cat(
        [to_blocked_mask[:, 1:-3], to_blocked_mask[:, 2:-2], to_blocked_mask[:, 3:-1]],
        dim=2,
    )
    # band_mask = tf.einsum(
    #     "BLQ,BLK->BLQK", from_blocked_mask[:, 2:-2], exp_blocked_to_pad
    # )
    band_mask = torch.einsum(
        "BLQ,BLK->BLQK", from_blocked_mask[:, 2:-2], exp_blocked_to_pad
    )
    band_mask = band_mask.unsqueeze(1)
    return band_mask


def create_attention_mask_from_input_mask(from_mask, to_mask):
    """Create attention mask from a 2D tensor mask.

    Args:
      from_mask: float32 Tensor of shape [batch_size, from_seq_length].
      to_mask: float32 Tensor of shape [batch_size, to_seq_length].

    Returns:
      float32 Tensor of shape [batch_size, 1, from_seq_length, to_seq_length].
    """
    # mask = tf.einsum("BF,BT->BFT", from_mask, to_mask)
    mask = torch.einsum("BF,BT->BFT", from_mask, to_mask)

    # expand to create a slot for heads.
    mask = mask.unsqueeze(1)

    return mask


class Dense3dLayer(nn.Module):
    """A dense layer with 3D kernel."""

    def __init__(
        self,
        num_attention_heads,
        size_per_head,
        activation,
        name=None,
        head_first=False,
        use_bias=True,
    ):
        """Constructor for dense layer with 3D kernel.

        Args:
          num_attention_heads: The size of output dimension.
          size_per_head: The size per attention head.
          initializer: Kernel initializer.
          activation: Actication function.
          name: The name scope of this layer.
          head_first: Whether to output head dimension before or after sequence dim.
          use_bias: Whether the layer uses a bias vector.
        """
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.size_per_head = size_per_head
        self.activation = activation
        self.head_first = head_first
        self.use_bias = use_bias

        hidden_size = self.num_attention_heads * self.size_per_head
        self.w = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.w = nn.init.trunc_normal_(self.w, std=0.02)

        if self.use_bias:
            self.b = nn.Parameter(torch.zeros(hidden_size))
        else:
            self.b = None

    def forward(self, input_tensor):
        """Constructor for dense layer with 3D kernel.

        Args:
          input_tensor: float Tensor of shape [batch, seq_length, hidden_size].

        Returns:
          float logits Tensor.
        """
        hidden_size = self.num_attention_heads * self.size_per_head
        reshape_w = self.w.reshape(
            hidden_size, self.num_attention_heads, self.size_per_head
        )
        if self.head_first:
            ret = torch.einsum("abc,cde->adbe", input_tensor, reshape_w)
        else:
            ret = torch.einsum("abc,cde->abde", input_tensor, reshape_w)

        if self.use_bias:
            if self.head_first:
                reshape_b = self.b.reshape(1, self.num_attention_heads, 1, self.size_per_head)
            else:
                reshape_b = self.b.reshape(self.num_attention_heads, self.size_per_head)
            ret += reshape_b
            
        if self.activation is not None:
            return self.activation(ret)
        else:
            return ret


class BigBird(nn.Module):
    def __init__(
        self,
        dim,
        heads,
        dim_head,
        num_rand_blocks=1,
        from_seq_length=64,
        to_seq_length=64,
        from_block_size=4,
        to_block_size=4,
        attention_probs_dropout_prob=0.0,
        use_bias=False,
        seed=None,
        query_act=None,
        key_act=None,
        value_act=None,
    ):
        super().__init__()
        self.num_attention_heads = heads
        self.size_per_head = dim_head
        self.num_rand_blocks = num_rand_blocks
        self.from_seq_length = from_seq_length
        self.to_seq_length = to_seq_length
        self.from_block_size = from_block_size
        self.to_block_size = to_block_size
        self.seed = seed
        self.dim = dim

        self.query_layer = Dense3dLayer(
            heads,
            dim_head,
            query_act,
            head_first=True,
            use_bias=use_bias,
        )

        self.key_layer = Dense3dLayer(
            heads,
            dim_head,
            key_act,
            head_first=True,
            use_bias=use_bias,
        )

        self.value_layer = Dense3dLayer(
            heads,
            dim_head,
            value_act,
            head_first=True,
            use_bias=use_bias,
        )

        self.attention_dropout = nn.Dropout(attention_probs_dropout_prob)
        self.rand_attn = self.generate_rand_attn_list()
        self.rand_block_mask = self.convert_attn_list_to_mask(self.rand_attn)
        self.attn_impl = self.bigbird_simulated_attention

    def generate_rand_attn_list(self):
        if self.seed is not None:
            np.random.seed(self.seed)
        if self.from_seq_length in [1024, 2048, 3072, 4096]:
            rand_attn = [
                bigbird_block_rand_mask(
                    MAX_SEQ_LEN,
                    MAX_SEQ_LEN,
                    self.from_block_size,
                    self.to_block_size,
                    self.num_rand_blocks,
                    last_idx=1024,
                )[: (self.from_seq_length // self.from_block_size - 2)]
                for _ in range(self.num_attention_heads)
            ]
        else:
            plan_from_length, plan_num_rand_blocks = get_rand_attn_plan(
                self.from_seq_length, self.from_block_size, self.num_rand_blocks
            )
            rand_attn = bigbird_block_rand_mask_with_head(
                seq_length=self.from_seq_length,
                block_size=self.from_block_size,
                num_heads=self.num_attention_heads,
                plan_from_length=plan_from_length,
                plan_num_rand_blocks=plan_num_rand_blocks,
            )
        rand_attn = np.stack(rand_attn, axis=0)
        return torch.tensor(rand_attn, dtype=torch.int32)

    def convert_attn_list_to_mask(self, rand_attn):
        temp_mask = [
            full_bigbird_mask(
                self.from_seq_length,
                self.to_seq_length,
                self.from_block_size,
                self.to_block_size,
                rand_attn=rand_attn[i],
            )
            for i in range(self.num_attention_heads)
        ]
        temp_mask = np.stack(temp_mask, axis=0)
        temp_mask = np.array(temp_mask, dtype=bool)
        rand_block_mask = torch.tensor(temp_mask, dtype=torch.bool)
        return rand_block_mask.float()

    def original_full_attention(
        self, query_layer, key_layer, value_layer, masks, training=True
    ):
        attention_mask = masks[0]
        attention_scores = torch.einsum("BNFH,BNTH->BNFT", query_layer, key_layer)
        attention_scores = attention_scores * (1.0 / np.sqrt(float(self.size_per_head)))

        if attention_mask is not None:
            adder = (1.0 - attention_mask) * -10000.0
            attention_scores += adder.to(attention_scores.device)

        attention_probs = torch.nn.functional.softmax(attention_scores, dim=-1)
        attention_probs = self.attention_dropout(attention_probs)

        context_layer = torch.einsum("BNFT,BNTH->BFNH", attention_probs, value_layer)
        return context_layer

    def bigbird_simulated_attention(
        self, query_layer, key_layer, value_layer, masks, training=True
    ):
        attention_mask = masks[0]
        rand_block_mask = self.rand_block_mask.unsqueeze(0)
        if attention_mask is not None:
            attention_mask = torch.minimum(attention_mask, rand_block_mask)
        else:
            attention_mask = rand_block_mask
        return self.original_full_attention(
            query_layer, key_layer, value_layer, [attention_mask], training=training
        )

    def forward(self, from_tensor, additional_args=None, training=True):
        to_tensor = from_tensor

        query = self.query_layer(from_tensor)
        key = self.key_layer(to_tensor)
        value = self.value_layer(to_tensor)

        masks = [None]  # Provide a default value

        contextual_output = self.attn_impl(query, key, value, masks, training=training)
        tensor_shape = contextual_output.shape
        contextual_output = contextual_output.reshape(tensor_shape[0], tensor_shape[1], -1)
        return contextual_output


if __name__ == "__main__":
    # Example usage
    attention = BigBird(dim=64, heads=16, dim_head=4).to('cuda:0')
    x = torch.randn(
        128, 64, 64
    ).to('cuda:0')  # Batch of 10 sequences of length 20 with dimension 128
    output = attention(x)
    print(output.shape)  # Should be (10, 20, 128)
