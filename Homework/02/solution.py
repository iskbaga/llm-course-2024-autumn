import torch
import torch.nn.functional as F
from pyarrow import float32


def compute_attention(queries, keys, values) -> torch.Tensor:
    """
    queries- (BATCH_SIZE, SEQ_LENGTH, HIDDEN_DIM)
    keys- (BATCH_SIZE, SEQ_LENGTH, HIDDEN_DIM)
    values- (BATCH_SIZE, SEQ_LENGTH, HIDDEN_DIM)
    """
    scores = torch.matmul(queries, keys.transpose(-2, -1)) / torch.sqrt(
        torch.tensor(keys.size(-1), dtype=torch.float32))

    weights = F.softmax(scores, dim=-1)

    outputs = torch.matmul(weights, values)

    return outputs


def compute_multihead_attention(queries, keys, values, projection_matrix) -> torch.Tensor:
    """
    queries- (BATCH_SIZE, N_HEADS, SEQ_LENGTH, DIM_PER_HEAD)
    keys- (BATCH_SIZE, N_HEADS, SEQ_LENGTH, DIM_PER_HEAD)
    values- (BATCH_SIZE, N_HEADS, SEQ_LENGTH, DIM_PER_HEAD)
    projection_matrix- (N_HEADS*DIM_PER_HEAD, N_HEADS*DIM_PER_HEAD)
    """
    batch_size, n_heads, seq_length, dim_per_head = queries.size()

    concatenated_outputs = torch.empty((batch_size, seq_length, 0), device=queries.device)

    for i in range(n_heads):
        query_i = queries[:, i, :, :]
        key_i = keys[:, i, :, :]
        value_i = values[:, i, :, :]
        output_i = compute_attention(query_i, key_i, value_i)
        concatenated_outputs = torch.cat((concatenated_outputs, output_i), dim=-1)

    projected_outputs = F.linear(concatenated_outputs, projection_matrix)
    return projected_outputs


def compute_rotary_embeddings(x: torch.Tensor) -> torch.Tensor:
    """
    Apply rotary embeddings to the input tensor.

    x - (BATCH_SIZE, SEQ_LENGTH, N_HEADS, DIM_PER_HEAD)

    Returns:
    - A tensor of shape (BATCH_SIZE, SEQ_LENGTH, N_HEADS, DIM_PER_HEAD)
      with rotary embeddings applied.
    """

    pass
