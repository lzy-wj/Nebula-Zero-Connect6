"""Raw-board TensorRT wrappers shared by inference-first candidates."""

import math

import torch
from torch import nn
from torch.nn import functional as F


class NativePairHeads(nn.Module):
    """Trainable low-rank pair policy and vector conditional value heads."""

    def __init__(self, feature_dim, pair_rank=16, board_size=19):
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.pair_rank = int(pair_rank)
        self.board_size = int(board_size)
        self.candidate_projection = nn.Linear(
            self.feature_dim,
            self.pair_rank,
            bias=False,
        )
        self.first_projection = nn.Linear(
            self.feature_dim,
            self.pair_rank,
            bias=False,
        )
        pair_value_hidden = max(64, self.feature_dim // 2)
        self.global_projection = nn.Linear(
            self.feature_dim,
            pair_value_hidden,
        )
        self.action_projection = nn.Linear(
            self.feature_dim,
            pair_value_hidden,
            bias=False,
        )
        self.value_output = nn.Linear(pair_value_hidden, 1)
        self.base_scale = nn.Parameter(torch.ones(()))
        relative_count = (self.board_size * 2 - 1) ** 2
        self.relative_bias = nn.Parameter(torch.zeros(relative_count))

        positions = torch.arange(self.board_size * self.board_size)
        self.register_buffer(
            "candidate_rows",
            positions // self.board_size,
            persistent=False,
        )
        self.register_buffer(
            "candidate_columns",
            positions % self.board_size,
            persistent=False,
        )
        nn.init.normal_(self.candidate_projection.weight, std=0.02)
        nn.init.normal_(self.first_projection.weight, std=0.02)
        nn.init.zeros_(self.value_output.weight)
        nn.init.zeros_(self.value_output.bias)

    def normalize(self, features):
        return F.layer_norm(features, (self.feature_dim,))

    def forward_all_from_normalized(self, normalized, parent_value):
        candidate_factors = self.candidate_projection(normalized)
        first_factors = self.first_projection(normalized)
        global_features = normalized.mean(dim=1)
        hidden = (
            self.global_projection(global_features).unsqueeze(1)
            + self.action_projection(normalized)
        )
        residual = self.value_output(F.gelu(hidden)).squeeze(2)
        parent_value = parent_value.float().flatten().clamp(-0.999, 0.999)
        parent_logit = 0.5 * (
            torch.log1p(parent_value) - torch.log1p(-parent_value)
        )
        pair_value = torch.tanh(parent_logit.unsqueeze(1) + residual.float())
        return candidate_factors, first_factors, pair_value

    def forward_all(self, features, parent_value):
        normalized = self.normalize(features)
        outputs = self.forward_all_from_normalized(normalized, parent_value)
        return (*outputs, normalized)

    def relative_indices(self, first_moves):
        first_rows = first_moves // self.board_size
        first_columns = first_moves % self.board_size
        relative_rows = self.candidate_rows.unsqueeze(0) - first_rows.unsqueeze(1)
        relative_columns = (
            self.candidate_columns.unsqueeze(0) - first_columns.unsqueeze(1)
        )
        relative_size = self.board_size * 2 - 1
        return (
            (relative_rows + self.board_size - 1) * relative_size
            + relative_columns
            + self.board_size
            - 1
        )

    def conditional_outputs(
        self,
        parent_policy_logits,
        candidate_factors,
        first_factors,
        pair_values,
        first_moves,
    ):
        batch_indices = torch.arange(
            first_moves.shape[0],
            device=first_moves.device,
        )
        selected_first = first_factors[batch_indices, first_moves]
        compatibility = (
            candidate_factors * selected_first.unsqueeze(1)
        ).sum(dim=2) / math.sqrt(self.pair_rank)
        relative_index = self.relative_indices(first_moves)
        second_logits = (
            self.base_scale * parent_policy_logits
            + compatibility
            + self.relative_bias[relative_index]
        )
        conditional_value = pair_values[batch_indices, first_moves]
        return second_logits, conditional_value


class FusedPairSelfPlayWrapper(nn.Module):
    """Fuse board encoding, rules and native low-rank pair outputs."""

    def __init__(
        self,
        model,
        compute_dtype=torch.float16,
        pair_rank=16,
        pair_heads=None,
    ):
        super().__init__()
        self.model = model
        self.compute_dtype = compute_dtype
        self.pair_rank = int(pair_rank)
        feature_dim = int(model.feature_dim)
        self.pair_heads = pair_heads or NativePairHeads(
            feature_dim,
            pair_rank=self.pair_rank,
        )
        if self.pair_heads.pair_rank != self.pair_rank:
            raise ValueError("pair head rank does not match wrapper rank")

        kernels = torch.zeros((4, 1, 6, 6), dtype=torch.float32)
        kernels[0, 0, 0, :] = 1
        kernels[1, 0, :, 0] = 1
        kernels[2, 0] = torch.eye(6)
        kernels[3, 0] = torch.rot90(torch.eye(6), 1, [0, 1])
        self.register_buffer("threat_kernels", kernels)

    def _detect_threats(self, stones, blockers):
        padded_stones = F.pad(stones, (5, 5, 5, 5), value=0)
        counts = F.conv2d(padded_stones, self.threat_kernels)
        padded_blockers = F.pad(blockers, (5, 5, 5, 5), value=1)
        blocked = F.conv2d(padded_blockers, self.threat_kernels)
        counts = counts * blocked.eq(0).float()
        return (
            counts.ge(6).float().flatten(1).amax(1),
            counts.eq(5).float().flatten(1).amax(1),
            counts.eq(4).float().flatten(1).amax(1),
        )

    def forward(self, board):
        board = torch.where(board == 2, -torch.ones_like(board), board)
        stone_count = board.ne(0).sum(dim=(1, 2))
        rank = (stone_count + 1) // 2
        current_player = torch.where(rank.remainder(2).eq(1), -1, 1)
        player_view = current_player.view(-1, 1, 1)

        self_stones = board.eq(player_view).float().unsqueeze(1)
        opponent_stones = board.eq(-player_view).float().unsqueeze(1)
        empty = board.eq(0).float().unsqueeze(1)
        black_plane = current_player.eq(1).float().view(
            -1, 1, 1, 1
        ).expand(-1, 1, 19, 19)
        second_stone = (
            stone_count.gt(0) & stone_count.remainder(2).eq(0)
        ).float().view(-1, 1, 1, 1).expand(-1, 1, 19, 19)
        inputs = torch.cat(
            (self_stones, opponent_stones, empty, black_plane, second_stone),
            dim=1,
        ).to(self.compute_dtype)

        policy_logits, _, raw_value, features = self.model(
            inputs,
            return_features=True,
        )
        raw_value = raw_value.flatten()
        normalized = self.pair_heads.normalize(features)
        candidate_factors, first_factors, pair_value = (
            self.pair_heads.forward_all_from_normalized(
                normalized,
                raw_value,
            )
        )

        policy_logits = policy_logits.float().masked_fill(
            empty.flatten(1).eq(0),
            -10_000.0,
        )
        policy = torch.softmax(policy_logits, dim=1)
        value = raw_value.float()

        my_win, my_five, my_four = self._detect_threats(
            self_stones,
            opponent_stones,
        )
        opponent_win, opponent_five, opponent_four = self._detect_threats(
            opponent_stones,
            self_stones,
        )
        stones_remaining = torch.where(
            stone_count.gt(0) & stone_count.remainder(2).eq(1),
            2,
            1,
        )
        value = torch.where(opponent_four.bool(), value.clamp(max=-0.2), value)
        value = torch.where(opponent_five.bool(), value.clamp(max=-0.25), value)
        can_win = my_five.bool() | (my_four.bool() & stones_remaining.ge(2))
        value = torch.where(can_win, torch.ones_like(value), value)
        value = torch.where(opponent_win.bool(), -torch.ones_like(value), value)
        value = torch.where(my_win.bool(), torch.ones_like(value), value)

        return (
            policy,
            value.unsqueeze(1),
            candidate_factors,
            first_factors,
            pair_value.to(self.compute_dtype),
        )
